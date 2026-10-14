"""Fish Audio Drama 3 nodes: load the model and codec, encode, save and load voice references, generate codes, decode
them to audio; and Qwen3-ASR to transcribe a reference.

The codec runs before and after generation only, so it is never resident next to the model while frames are sampled.
"""
import os

import torch
from safetensors.torch import save_file

import comfy.audio
import comfy.model_management
import comfy.utils
import folder_paths
from comfy_api.latest import ComfyExtension, io

from .drama3.chunking import split_text
from .drama3.codec import FRAME_LENGTH, SAMPLE_RATE
from .drama3.generate import generate
from .drama3.loader import load_codec, load_drama3
from .drama3.prompt import build_prompt
from .qwen3_asr.asr import LANGUAGES, load_qwen3_asr, transcribe

Drama3Model = io.Custom("DRAMA3_MODEL")
Drama3Codec = io.Custom("DRAMA3_CODEC")
Drama3Voice = io.Custom("DRAMA3_VOICE")
Drama3Codes = io.Custom("DRAMA3_CODES")
Qwen3ASRModel = io.Custom("QWEN3_ASR_MODEL")
# the vendor's limit on prompt rows (references, history and text)
MAX_PROMPT_ROWS = 8192


VOICES_FOLDER = os.path.join(folder_paths.get_output_directory(), "drama3_voices")


def register_folder(name, path):
    """`path` after any paths extra_model_paths.yaml gives, listed with the usual model file extensions."""
    folder_paths.add_model_folder_path(name, path)
    folder_paths.folder_names_and_paths[name] = (folder_paths.folder_names_and_paths[name][0], folder_paths.supported_pt_extensions)


register_folder("asr", os.path.join(folder_paths.models_dir, "asr"))
register_folder("drama3_voices", VOICES_FOLDER)


def gpu_choices():
    return ["none"] + [f"cuda:{i}" for i in range(torch.cuda.device_count())]


class Drama3Loader(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="Drama3Loader",
            display_name="Load Drama 3",
            category="audio/drama3",
            description="Fish Audio Drama 3, converted with the pack's tools/convert.py (int8 ConvRot: 36 GB).",
            inputs=[
                io.Combo.Input("model_name", options=folder_paths.get_filename_list("diffusion_models")),
                io.Combo.Input("experts", options=["in RAM", "in VRAM"], default="in RAM",
                               tooltip="Where the 256 x 40 routed experts (90% of the model) live. In RAM, each frame copies only the "
                                       "8 experts per layer it uses and keeps recent ones in a VRAM cache: needs the file's size in "
                                       "system RAM. In VRAM needs the whole file in VRAM (or split over two GPUs)."),
                io.Float.Input("expert_cache_gb", default=0.0, min=0.0, max=256.0, step=0.5,
                               tooltip="VRAM for cached experts when they live in RAM; 0 uses what is free when generation starts."),
                io.Combo.Input("second_gpu", options=gpu_choices(), default="none", advanced=True,
                               tooltip="Run the last backbone layers, the head and the tower on this GPU. The int8 file fits two "
                                       "24 GB cards with the experts in VRAM."),
                io.Int.Input("split_layer", default=0, min=0, max=39, advanced=True,
                             tooltip="First layer on the second GPU; 0 balances the two cards' free memory."),
            ],
            outputs=[Drama3Model.Output()],
        )

    @classmethod
    def execute(cls, model_name, experts="in RAM", expert_cache_gb=0.0, second_gpu="none", split_layer=0) -> io.NodeOutput:
        path = folder_paths.get_full_path_or_raise("diffusion_models", model_name)
        second = None if second_gpu == "none" else torch.device(second_gpu)
        if second is not None and second == comfy.model_management.get_torch_device():
            raise ValueError(f"Drama 3: the second GPU must differ from ComfyUI's main device {second}")
        return io.NodeOutput(load_drama3(path, second, split_layer, experts_in_ram=experts == "in RAM", expert_cache_gb=expert_cache_gb))


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
            description="A voice to clone: a clip and its exact transcript. A few seconds of clean speech work best.",
            inputs=[
                Drama3Codec.Input("codec"),
                io.Audio.Input("audio"),
                io.String.Input("transcript", multiline=True, tooltip="What the clip says, word for word (Qwen3-ASR Transcribe can "
                                "write it). A leading <|speaker:N|> tag assigns the voice to speaker N (0 if absent)."),
            ],
            outputs=[Drama3Voice.Output()],
        )

    @classmethod
    def execute(cls, codec, audio, transcript) -> io.NodeOutput:
        comfy.model_management.load_models_gpu([codec.patcher])
        device = codec.patcher.load_device
        waveform = audio["waveform"][0].float().mean(0, keepdim=True)
        if audio["sample_rate"] != SAMPLE_RATE:
            waveform = comfy.audio.resample(waveform, audio["sample_rate"], SAMPLE_RATE)
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
                        "<|speaker:N|> switches between the voices' speakers.",
            inputs=[
                Drama3Model.Input("model"),
                io.String.Input("text", multiline=True),
                io.Autogrow.Input("voices", template=voices),
                io.Int.Input("seed", default=0, min=0, max=0xffffffffffffffff, control_after_generate=True),
                io.Float.Input("temperature", default=1.0, min=0.0, max=2.0, step=0.05,
                               tooltip="1.0 with top_p 0.9 and top_k 30 are the defaults of Fish Audio's Drama 3 server; "
                                       "lower values make takes fall silent more often."),
                io.Float.Input("top_p", default=0.9, min=0.05, max=1.0, step=0.05),
                io.Int.Input("top_k", default=30, min=1, max=4096),
                io.Int.Input("max_frames", default=600, min=1, max=8192,
                             tooltip="At most this many frames per text chunk (21.5 per second of audio); generation ends earlier "
                                     "at end of speech."),
                io.Boolean.Input("split_long_text", default=True,
                                 tooltip="Generate long text in chunks of about 300 bytes, each continuing from the first and the "
                                         "latest chunk, as Fish Audio's own server does."),
            ],
            outputs=[Drama3Codes.Output()],
        )

    @classmethod
    def execute(cls, model, text, voices, seed, temperature, top_p, top_k, max_frames, split_long_text=True) -> io.NodeOutput:
        references = [(voice["transcript"], voice["codes"]) for voice in voices.values() if voice is not None]
        chunks = (split_text(text) if split_long_text else [text]) or [text]
        comfy.model_management.load_models_gpu(model.patchers())
        progress = comfy.utils.ProgressBar(max_frames * len(chunks))
        first = latest = None
        frames = []
        for index, chunk in enumerate(chunks):
            prompt = chunk_prompt(model, chunk, references, [h for h in (first, latest) if h is not None])

            def on_frame(step, done=index * max_frames):
                comfy.model_management.throw_exception_if_processing_interrupted()
                progress.update_absolute(done + step + 1)

            chunk_frames = generate(model, prompt, max_frames, temperature, top_k, top_p, seed + index, on_frame)
            frames.append(chunk_frames)
            turn = (chunk, chunk_frames[:, 1:].T.contiguous())
            if first is None:
                first = turn
            else:
                latest = turn
        return io.NodeOutput(torch.cat(frames)[:, 1:].T.contiguous())


def chunk_prompt(model, text, references, history):
    """The prompt for one chunk with as much history as fits: the latest chunk goes first, then the first one."""
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
            inputs=[Drama3Codec.Input("codec"), Drama3Codes.Input("codes")],
            outputs=[io.Audio.Output()],
        )

    @classmethod
    def execute(cls, codec, codes) -> io.NodeOutput:
        comfy.model_management.load_models_gpu([codec.patcher])
        device = codec.patcher.load_device
        if codes.shape[-1] == 0:
            waveform = torch.zeros((1, 1, FRAME_LENGTH))
        else:
            with comfy.model_management.cuda_device_context(device):
                waveform = codec.model.decode(codes[None].to(device), torch.float32).float().cpu()[:, None]
        return io.NodeOutput({"waveform": waveform, "sample_rate": SAMPLE_RATE})


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
        return [Drama3Loader, Drama3CodecLoader, Drama3EncodeVoice, Drama3SaveVoice, Drama3LoadVoice, Drama3Generate, Drama3Decode,
                Qwen3ASRLoader, Qwen3ASRTranscribe]


async def comfy_entrypoint() -> Drama3Extension:
    return Drama3Extension()
