"""Fish Audio Drama 3 nodes: load the model and codec, encode, save and load voice references, generate codes, decode
them to audio; and Qwen3-ASR to transcribe a reference.

The codec runs before and after generation only, so it is never resident next to the model while frames are sampled.
"""
import hashlib
import json
import logging
import math
import os
import wave

import torch
from aiohttp import web
from safetensors.torch import save_file

import comfy.audio
import comfy.model_management
import comfy.utils
import folder_paths
from comfy_api.latest import ComfyExtension, io
from server import PromptServer

from .drama3.audio import tidy_clip, trim_pauses
from .drama3.chunking import CHUNK_BYTES, CHUNK_MAX_SPEAKERS, by_speaker, chunk_lines, split_text
from .drama3.editor import DEFAULTS, analyze, parse_edits, plan, render
from .drama3.codec import DECODE_CONTEXT, DECODE_FRAME_BYTES, DECODE_WINDOW, FRAME_LENGTH, SAMPLE_RATE
from .drama3.generate import generate
from .drama3.loader import load_codec, load_drama3
from .drama3.prompt import LINE_SEPARATOR, MAX_SPEAKER, as_speaker_0, build_prompt, speaker_of, with_speaker_tag
from .qwen3_asr.asr import LANGUAGES, load_qwen3_asr, transcribe

Drama3Model = io.Custom("DRAMA3_MODEL")
Drama3Codec = io.Custom("DRAMA3_CODEC")
Drama3Voice = io.Custom("DRAMA3_VOICE")
Drama3Codes = io.Custom("DRAMA3_CODES")
Qwen3ASRModel = io.Custom("QWEN3_ASR_MODEL")
# the vendor's limit on prompt rows (references, history and text)
MAX_PROMPT_ROWS = 8192
# seconds of silence Decode puts between separately generated takes, the Editor's shortest gap between lines
TAKE_GAP = 0.3
# Fish Audio's Drama 3 Space shortens longer pauses to this many seconds
MAX_PAUSE = 0.75
SPEAKER_MODES = ["one take per line", "one take per line, hearing the line before", "one take per speaker", "together"]
EXPERTS_SPLIT = "split VRAM + RAM"


VOICES_FOLDER = os.path.join(folder_paths.get_output_directory(), "drama3_voices")
CODES_FOLDER = os.path.join(folder_paths.get_output_directory(), "drama3_codes")


def register_folder(name, path):
    """`path` after any paths extra_model_paths.yaml gives, listed with the usual model file extensions."""
    folder_paths.add_model_folder_path(name, path)
    folder_paths.folder_names_and_paths[name] = (folder_paths.folder_names_and_paths[name][0], folder_paths.supported_pt_extensions)


register_folder("asr", os.path.join(folder_paths.models_dir, "asr"))
register_folder("drama3_voices", VOICES_FOLDER)
register_folder("drama3_codes", CODES_FOLDER)


def gpu_choices():
    return ["none"] + [f"cuda:{i}" for i in range(torch.cuda.device_count())]


class Drama3Loader(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="Drama3Loader",
            display_name="Load Drama 3",
            category="audio/drama3",
            description="Fish Audio Drama 3, converted with the pack's tools/convert.py (int8 ConvRot: 36 GB). Split VRAM + RAM "
                        "keeps the first layers' experts on the GPU and only the others' in RAM, for systems whose RAM cannot "
                        "hold the file.",
            inputs=[
                io.Combo.Input("model_name", options=folder_paths.get_filename_list("diffusion_models")),
                io.Combo.Input("experts", options=["in RAM", "in VRAM", EXPERTS_SPLIT], default="in RAM",
                               tooltip="Where the 256 x 40 routed experts (90% of the model) live. In RAM, each frame copies only the "
                                       "8 experts per layer it uses and keeps recent ones in a VRAM cache: needs the file's size in "
                                       "system RAM. In VRAM needs the whole file in VRAM (or split over two GPUs). Split VRAM + RAM "
                                       "keeps the experts of the first vram_expert_layers layers in VRAM and the others' in RAM, "
                                       "behind the cache: needs only those in RAM."),
                io.Float.Input("expert_cache_gb", default=0.0, min=0.0, max=256.0, step=0.5,
                               tooltip="VRAM for cached experts when they live in RAM (all or split); 0 uses what is free when "
                                       "generation starts."),
                io.Combo.Input("second_gpu", options=gpu_choices(), default="none", advanced=True,
                               tooltip="Run the last backbone layers, the head and the tower on this GPU. The int8 file fits two "
                                       "24 GB cards with the experts in VRAM."),
                io.Int.Input("split_layer", default=0, min=0, max=39, advanced=True,
                             tooltip="First layer on the second GPU; 0 balances the two cards' free memory."),
                # last, so that workflows saved before it keep their values in place
                io.Int.Input("vram_expert_layers", default=0, min=0, max=40,
                             tooltip="With the experts split: how many layers (the first ones) keep their experts in VRAM, "
                                     "0.75 GiB each with int8 or FP8; the others' stay in RAM. 0 fits as many as the card holds "
                                     "next to the rest of the model (18 int8 layers on 24 GB). Fewer leave more VRAM to the "
                                     "cache, which is faster, but need more RAM."),
            ],
            outputs=[Drama3Model.Output()],
        )

    @classmethod
    def execute(cls, model_name, experts="in RAM", expert_cache_gb=0.0, second_gpu="none", split_layer=0,
                vram_expert_layers=0) -> io.NodeOutput:
        path = folder_paths.get_full_path_or_raise("diffusion_models", model_name)
        second = None if second_gpu == "none" else torch.device(second_gpu)
        if second is not None and second == comfy.model_management.get_torch_device():
            raise ValueError(f"Drama 3: the second GPU must differ from ComfyUI's main device {second}")
        return io.NodeOutput(load_drama3(path, second, split_layer, experts_in_ram=experts == "in RAM", expert_cache_gb=expert_cache_gb,
                                         split_experts=experts == EXPERTS_SPLIT, vram_layers=vram_expert_layers))


class Drama3CodecLoader(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="Drama3CodecLoader",
            display_name="Load Drama 3 Codec",
            category="audio/drama3",
            description="The Fish S2-Pro codec Drama 3 speaks in (tools/convert.py codec): 44.1 kHz, 21.5 frames/s.",
            inputs=[io.Combo.Input("codec_name", options=folder_paths.get_filename_list("vae"))],
            outputs=[Drama3Codec.Output()],
        )

    @classmethod
    def execute(cls, codec_name) -> io.NodeOutput:
        return io.NodeOutput(load_codec(folder_paths.get_full_path_or_raise("vae", codec_name)))


class Drama3EncodeVoice(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="Drama3EncodeVoice",
            display_name="Drama 3 Encode Voice",
            category="audio/drama3",
            description="A voice to clone: a clip and its exact transcript. A few seconds of clean speech work best; the clip "
                        "loses its silent edges and is peak-normalized, as Fish Audio's Drama 3 Space does.",
            inputs=[
                Drama3Codec.Input("codec"),
                io.Audio.Input("audio"),
                io.String.Input("transcript", multiline=True, tooltip="What the clip says, word for word (Qwen3-ASR Transcribe can "
                                "write it). The voice speaks for the speaker of its slot on Drama 3 Generate (voice1 is "
                                "<|speaker:1|>); a leading <|speaker:N|> tag here picks speaker N instead."),
            ],
            outputs=[Drama3Voice.Output()],
        )

    @classmethod
    def execute(cls, codec, audio, transcript) -> io.NodeOutput:
        # whole: a partly loaded codec leaves its norms' weights on the CPU
        comfy.model_management.load_models_gpu([codec.patcher], force_full_load=True)
        device = codec.patcher.load_device
        waveform = audio["waveform"][0].float().mean(0, keepdim=True)
        if audio["sample_rate"] != SAMPLE_RATE:
            waveform = comfy.audio.resample(waveform, audio["sample_rate"], SAMPLE_RATE)
        waveform = tidy_clip(waveform)
        with comfy.model_management.cuda_device_context(device):
            codes = codec.model.encode(waveform.to(device))[0].cpu()
        return io.NodeOutput({"transcript": transcript, "codes": codes})


class Drama3SaveVoice(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="Drama3SaveVoice",
            display_name="Save Drama 3 Voice",
            category="audio/drama3",
            description="Save an encoded voice (its codes and transcript) to output/drama3_voices, for Load Drama 3 Voice.",
            inputs=[
                Drama3Voice.Input("voice"),
                io.String.Input("name", default="voice", tooltip="File name prefix; a counter is added. Subfolders are allowed."),
            ],
            is_output_node=True,
        )

    @classmethod
    def execute(cls, voice, name) -> io.NodeOutput:
        folder, filename, counter, _, _ = folder_paths.get_save_image_path(name, VOICES_FOLDER)
        save_file({"codes": voice["codes"].to(torch.int16).contiguous()}, os.path.join(folder, f"{filename}_{counter:05}_.safetensors"),
                  metadata={"transcript": voice["transcript"]})
        return io.NodeOutput()


class Drama3LoadVoice(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="Drama3LoadVoice",
            display_name="Load Drama 3 Voice",
            category="audio/drama3",
            description="A voice saved by Save Drama 3 Voice (output/drama3_voices).",
            inputs=[io.Combo.Input("voice", options=folder_paths.get_filename_list("drama3_voices"))],
            outputs=[Drama3Voice.Output()],
        )

    @classmethod
    def execute(cls, voice) -> io.NodeOutput:
        tensors, metadata = comfy.utils.load_torch_file(folder_paths.get_full_path_or_raise("drama3_voices", voice), return_metadata=True)
        if not metadata or "transcript" not in metadata or "codes" not in tensors:
            raise ValueError(f"{voice} is not a Drama 3 voice file")
        return io.NodeOutput({"transcript": metadata["transcript"], "codes": tensors["codes"].long()})


class Drama3Generate(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        voices = io.Autogrow.TemplatePrefix(Drama3Voice.Input("voice"), prefix="voice", min=0, max=16)
        return io.Schema(
            node_id="Drama3Generate",
            display_name="Drama 3 Generate",
            category="audio/drama3",
            description="Speak the text. Tags such as [sighs] or [laughs] go where the sound should be; "
                        "<|speaker:N|> switches speakers, and the voice in slot voiceN speaks for speaker N.",
            inputs=[
                Drama3Model.Input("model"),
                io.String.Input("text", multiline=True),
                io.Autogrow.Input("voices", template=voices),
                io.Int.Input("seed", default=0, min=0, max=0xffffffffffffffff, control_after_generate=True),
                io.Float.Input("temperature", default=1.0, min=0.0, max=2.0, step=0.05,
                               tooltip="1.0 with top_p 0.9 are the defaults of Fish Audio's Drama 3 server and Space; "
                                       "lower values make takes fall silent more often."),
                io.Float.Input("top_p", default=0.9, min=0.05, max=1.0, step=0.05),
                io.Int.Input("top_k", default=0, min=0, max=4096,
                             tooltip="0 keeps every candidate, as Fish Audio's Drama 3 Space does: at a pause a few silence codes "
                                     "outrank the many ways speech resumes, and top_k 30 left 11 of their 40 takes silent."),
                io.Int.Input("max_frames", default=600, min=1, max=8192,
                             tooltip="At most this many frames per text chunk (21.5 per second of audio); generation ends earlier "
                                     "at end of speech."),
                io.Boolean.Input("split_long_text", default=True,
                                 tooltip="Generate long text in chunks of about 300 bytes, each continuing from the first and the "
                                         "latest chunk, as Fish Audio's own server does."),
                io.Combo.Input("speakers", options=SPEAKER_MODES, default="one take per line",
                               tooltip="one take per line: each line is generated on its own, with only its speaker's voices and "
                                       "earlier takes in the prompt, as Fish Audio's Drama 3 Space voices a scene; voices cannot "
                                       "bleed into each other, but lines do not hear each other. hearing the line before: the same "
                                       "plus the take before it as context, so lines react again but that take's voice can bleed "
                                       "in. one take per speaker: all of a speaker's lines in one take, cut at the line breaks it "
                                       "says and put back in script order. together: the dialogue as one script, every line "
                                       "hearing the others. Decode and the Editor decode each take on its own."),
            ],
            outputs=[Drama3Codes.Output()],
        )

    @classmethod
    def execute(cls, model, text, voices, seed, temperature, top_p, top_k, max_frames, split_long_text=True,
                speakers="one take per line") -> io.NodeOutput:
        # an untagged transcript speaks for its slot: voice1 is <|speaker:1|>
        references = [(with_speaker_tag(voice["transcript"], int(name.removeprefix("voice"))), voice["codes"])
                      for name, voice in voices.items() if voice is not None]
        max_bytes = CHUNK_BYTES if split_long_text else math.inf
        together = speakers == "together"
        max_speakers = 1 if not together else CHUNK_MAX_SPEAKERS if split_long_text else MAX_SPEAKER + 1
        chunks = split_text(text, max_bytes, max_speakers) or [text]
        # one take per speaker: the script position of each take's lines
        positions = None
        if speakers == "one take per speaker":
            chunks, positions = by_speaker(chunks, max_bytes)
        # an earlier take of a dialogue hands the voice it ends in to the next chunk (6 of 16 lines in the wrong voice, 1 of
        # 18 without it): when every speaker has a voice, the references alone keep them apart
        speakers_in_text = {speaker_of(line) for chunk in chunks for line in chunk.split(LINE_SEPARATOR)}
        cast_dialogue = len(speakers_in_text) > 1 and speakers_in_text <= {speaker_of(transcript) for transcript, _ in references}
        comfy.model_management.load_models_gpu(model.patchers())
        progress = comfy.utils.ProgressBar(max_frames * len(chunks))
        config = model.config
        separator = model.tokenizer.token_to_id("<|audio_separator|>")
        # taken: (text, frames) of each take with sound; pieces: (script position, lines, codes) decoded as one each
        taken, pieces = [], []
        # the takes share one expert cache: each starts with what the earlier ones loaded
        with model.session():
            for index, chunk in enumerate(chunks):
                speaker = speaker_of(chunk)
                if together:
                    said, cast, own = chunk, references, [] if cast_dialogue else taken
                else:
                    # a solo take: the speaker's lines and voices as speaker 0
                    said = as_speaker_0(chunk, speaker)
                    cast = [(as_speaker_0(transcript, speaker), codes) for transcript, codes in references if speaker_of(transcript) == speaker]
                    own = [turn for turn in taken if speaker_of(turn[0]) == speaker]
                # the first and the latest take go back into each prompt; separately, those of the take's own speaker
                history = own[:1] + own[1:][-1:]
                if speakers == "one take per line, hearing the line before" and taken and speaker_of(taken[-1][0]) != speaker:
                    history.append(taken[-1])
                prompt = chunk_prompt(model, said, cast, [(turn if together else as_speaker_0(turn, speaker), frames) for turn, frames in history])

                def on_frame(step, done=index * max_frames):
                    comfy.model_management.throw_exception_if_processing_interrupted()
                    progress.update_absolute(done + step + 1)

                chunk_frames = generate(model, prompt, max_frames, temperature, top_k, top_p, seed + index, on_frame,
                                        lines=LINE_SEPARATOR in chunk)
                # the take's own line separators are text rows
                audio = chunk_frames[(chunk_frames[:, 0] >= config.semantic_start_token_id) & (chunk_frames[:, 0] <= config.semantic_end_token_id), 1:].T
                lines = chunk_lines(chunk, index, chunk_frames, 0, config, separator)
                if positions is None:
                    pieces.append((index, lines, audio))
                else:
                    if len(lines) < len(positions[index]):
                        logging.warning("Drama 3: speaker %d's take marked %d of its %d line breaks, so its last %d lines are joined or "
                                        "missing, out of script order; another seed or one take per line avoids this.", speaker,
                                        len(lines) - 1, len(positions[index]) - 1, len(positions[index]) - len(lines) + 1)
                    pieces += [(position, [{**line, "start": 0, "end": line["end"] - line["start"]}], audio[:, line["start"]:line["end"]])
                               for line, position in zip(lines, positions[index])]
                if len(chunk_frames):
                    # fish-speech leaves an empty take out of the history, and Fish's builders refuse zero-frame codes
                    taken.append((chunk, chunk_frames))
        pieces.sort(key=lambda piece: piece[0])
        lines, sound = [], 0
        for _, piece_lines, audio in pieces:
            lines += [{**line, "start": line["start"] + sound, "end": line["end"] + sound} for line in piece_lines]
            sound += audio.shape[-1]
        # takes: the frames decoded as one stream each; chunks of one script continue each other's audio
        takes = [sound] if together else [audio.shape[-1] for _, _, audio in pieces]
        return io.NodeOutput({"codes": torch.cat([audio for _, _, audio in pieces], -1).contiguous(), "lines": lines, "takes": takes})


class Drama3SaveCodes(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="Drama3SaveCodes",
            display_name="Save Drama 3 Codes",
            category="audio/drama3",
            description="Save a generated take (its codes and line timing, a few hundred KB) to output/drama3_codes, to decode "
                        "or edit it again with Load Drama 3 Codes without generating it again.",
            inputs=[
                Drama3Codes.Input("codes"),
                io.String.Input("name", default="drama3/take", tooltip="File name prefix; a counter is added. Subfolders are allowed."),
            ],
            is_output_node=True,
        )

    @classmethod
    def execute(cls, codes, name) -> io.NodeOutput:
        folder, filename, counter, _, _ = folder_paths.get_save_image_path(name, CODES_FOLDER)
        save_file({"codes": codes["codes"].to(torch.int16).contiguous()}, os.path.join(folder, f"{filename}_{counter:05}_.safetensors"),
                  metadata={"lines": json.dumps(codes["lines"]), "takes": json.dumps(codes["takes"])})
        return io.NodeOutput()


class Drama3LoadCodes(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="Drama3LoadCodes",
            display_name="Load Drama 3 Codes",
            category="audio/drama3",
            description="A take saved by Save Drama 3 Codes (output/drama3_codes), for Decode or the Editor.",
            inputs=[io.Combo.Input("codes", options=folder_paths.get_filename_list("drama3_codes"))],
            outputs=[Drama3Codes.Output()],
        )

    @classmethod
    def execute(cls, codes) -> io.NodeOutput:
        tensors, metadata = comfy.utils.load_torch_file(folder_paths.get_full_path_or_raise("drama3_codes", codes), return_metadata=True)
        if not metadata or "lines" not in metadata or "takes" not in metadata or "codes" not in tensors:
            raise ValueError(f"{codes} is not a Drama 3 codes file")
        return io.NodeOutput({"codes": tensors["codes"].long(), "lines": json.loads(metadata["lines"]), "takes": json.loads(metadata["takes"])})


def chunk_prompt(model, text, references, history):
    """The prompt for one chunk with as much history as fits: the latest chunk is dropped first, then the first one."""
    for turns in (history, history[:1], []):
        prompt = build_prompt(model.tokenizer, model.config, text, references, turns)
        if len(prompt) <= MAX_PROMPT_ROWS:
            return prompt
    raise ValueError(f"Drama 3: the voice references alone make a prompt of {len(prompt)} rows; at most {MAX_PROMPT_ROWS} fit")


class Drama3Decode(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="Drama3Decode",
            display_name="Drama 3 Decode",
            category="audio/drama3",
            description="Codes to 44.1 kHz audio. Speakers generated separately are decoded take by take and joined in script "
                        "order with 0.3 s of silence between takes.",
            inputs=[
                Drama3Codec.Input("codec"),
                Drama3Codes.Input("codes"),
                io.Boolean.Input("shorten_long_pauses", default=True,
                                 tooltip="Shorten pauses longer than 0.75 s to 0.75 s, cut the silence before the first sound and "
                                         "keep 0.15 s after the last, as Fish Audio's Drama 3 Space does. Off keeps the take as "
                                         "generated."),
            ],
            outputs=[io.Audio.Output()],
        )

    @classmethod
    def execute(cls, codec, codes, shorten_long_pauses=True) -> io.NodeOutput:
        waveform = decode(codec, codes, TAKE_GAP)
        if shorten_long_pauses and codes["codes"].shape[-1]:
            waveform = trim_pauses(waveform, SAMPLE_RATE, MAX_PAUSE)
        return io.NodeOutput({"waveform": waveform[None], "sample_rate": SAMPLE_RATE})


def decode(codec, codes, gap=0.0):
    """Audio [1, samples] of Generate's codes: each take decoded on its own, `gap` seconds of silence between takes, with
    room made in VRAM for one decode window."""
    takes = [take for take in codes["codes"].split(codes["takes"], -1) if take.shape[-1]]
    if not takes:
        return torch.zeros((1, FRAME_LENGTH))
    window = min(max(take.shape[-1] for take in takes), DECODE_CONTEXT + DECODE_WINDOW)
    # whole, making room by unloading Drama 3's weights: a partly loaded codec leaves its norms' weights on the CPU
    comfy.model_management.load_models_gpu([codec.patcher], memory_required=window * DECODE_FRAME_BYTES, force_full_load=True)
    device = codec.patcher.load_device
    with comfy.model_management.cuda_device_context(device):
        audio = [codec.model.decode(take[None].to(device), torch.float32).float().cpu() for take in takes]
    silence = torch.zeros((1, round(gap * SAMPLE_RATE)))
    return torch.cat([piece for take in audio for piece in (silence, take)][1:], -1)


class Drama3Editor(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="Drama3Editor",
            display_name="Drama 3 Editor",
            category="audio/drama3",
            is_output_node=True,
            search_aliases=["drama 3 timeline", "edit take", "remove silence", "shorten pauses"],
            description="Decode a take and edit it: captions follow the voice word by word, one track per voice. Edit opens "
                        "the editor over ComfyUI: choose how the silences sound, check the lines it flags, cut, add silence, "
                        "name voices and set their volume. Use these edits runs the node again with them; they stay in the "
                        "workflow.",
            inputs=[
                Drama3Codec.Input("codec"),
                Drama3Codes.Input("codes"),
                Qwen3ASRModel.Input("aligner", optional=True,
                                    tooltip="Qwen3-ForcedAligner from Load Qwen3-ASR: words light up as they are spoken. Without "
                                            "it, whole lines do."),
                io.String.Input("edits", default="", optional=True, socketless=True, extra_dict={"hidden": True},
                                tooltip="Settings and edits made in the editor (managed by the editor)."),
            ],
            outputs=[io.Audio.Output()],
        )

    @classmethod
    def execute(cls, codec, codes, aligner=None, edits="") -> io.NodeOutput:
        waveform = decode(codec, codes)[0]
        analysis = analyze(waveform, SAMPLE_RATE, codes["lines"], aligner)
        settings = parse_edits(edits)
        audio = render(waveform, analysis, plan(analysis, settings)["pieces"], settings)
        return io.NodeOutput({"waveform": audio[None, None], "sample_rate": SAMPLE_RATE}, ui={"drama3_editor": [save_preview(waveform, analysis)]})


def save_preview(waveform, analysis):
    """The unedited take (16-bit WAV) and its analysis (JSON) in the temp folder, for the editor to play and draw; the
    workflow keeps only their names."""
    pcm = (waveform.clamp(-1, 1) * 32767).round().to(torch.int16).numpy().tobytes()
    name = f"{hashlib.sha1(pcm).hexdigest()[:16]}{'_aligned' if analysis['aligned'] else ''}"
    folder = os.path.join(folder_paths.get_temp_directory(), "drama3_editor")
    os.makedirs(folder, exist_ok=True)
    with wave.open(os.path.join(folder, f"{name}.wav"), "wb") as file:
        file.setnchannels(1)
        file.setsampwidth(2)
        file.setframerate(SAMPLE_RATE)
        file.writeframes(pcm)
    with open(os.path.join(folder, f"{name}.json"), "w", encoding="utf-8") as file:
        json.dump(analysis, file)
    return {"audio": f"{name}.wav", "analysis": f"{name}.json", "subfolder": "drama3_editor", "type": "temp"}


if getattr(PromptServer, "instance", None) is not None:
    @PromptServer.instance.routes.post("/drama3/editor/plan")
    async def editor_plan(request):
        """The edit list the editor previews: the node's analysis and the editor's current settings and edits."""
        body = await request.json()
        return web.json_response(plan(body["analysis"], {**DEFAULTS, **body.get("edits", {})}))


class Qwen3ASRLoader(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="Qwen3ASRLoader",
            display_name="Load Qwen3-ASR",
            category="audio/qwen3_asr",
            description="Qwen3-ASR speech recognition (1.7B or 0.6B), converted with the pack's tools/convert.py asr (models/asr).",
            inputs=[io.Combo.Input("model_name", options=folder_paths.get_filename_list("asr"))],
            outputs=[Qwen3ASRModel.Output()],
        )

    @classmethod
    def execute(cls, model_name) -> io.NodeOutput:
        return io.NodeOutput(load_qwen3_asr(folder_paths.get_full_path_or_raise("asr", model_name)))


class Qwen3ASRTranscribe(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="Qwen3ASRTranscribe",
            display_name="Qwen3-ASR Transcribe",
            category="audio/qwen3_asr",
            description="Speech to text in 30 languages and 22 Chinese dialects, singing included. Its text can feed Drama 3 "
                        "Encode Voice's transcript.",
            inputs=[
                Qwen3ASRModel.Input("model"),
                io.Audio.Input("audio"),
                io.Combo.Input("language", options=["auto", *LANGUAGES], default="auto",
                               tooltip="auto detects the language; naming it writes the text without detecting."),
                io.String.Input("context", multiline=True, default="",
                                tooltip="Optional text that helps recognition: names, terms, the topic."),
                io.Int.Input("max_new_tokens", default=4096, min=1, max=32768, advanced=True,
                             tooltip="At most this many tokens per 20 minutes of audio."),
            ],
            outputs=[io.String.Output("text", display_name="text"), io.String.Output("language", display_name="language")],
        )

    @classmethod
    def execute(cls, model, audio, language="auto", context="", max_new_tokens=4096) -> io.NodeOutput:
        results = [transcribe(model, waveform, audio["sample_rate"], None if language == "auto" else language, context, max_new_tokens)
                   for waveform in audio["waveform"]]
        return io.NodeOutput("\n".join(text for text, _ in results), ",".join(dict.fromkeys(found for _, found in results if found)))


class Drama3Extension(ComfyExtension):
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [Drama3Loader, Drama3CodecLoader, Drama3EncodeVoice, Drama3SaveVoice, Drama3LoadVoice, Drama3Generate, Drama3SaveCodes,
                Drama3LoadCodes, Drama3Decode, Drama3Editor,
                Qwen3ASRLoader, Qwen3ASRTranscribe]


async def comfy_entrypoint() -> Drama3Extension:
    return Drama3Extension()
