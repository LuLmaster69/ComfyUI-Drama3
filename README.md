<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/banner-dark.png">
  <img alt="ComfyUI-Drama3: Fish Audio Drama 3 text to speech, natively in ComfyUI" src="assets/banner-light.png" width="100%">
</picture>

<p align="center">
  <img alt="ComfyUI v0.40+" src="https://img.shields.io/badge/ComfyUI-v0.40%2B-FFA931?style=flat-square&labelColor=555">
  <img alt="VRAM 8 GB and up" src="https://img.shields.io/badge/VRAM-8%20GB%2B-FFA931?style=flat-square&labelColor=555">
  <img alt="RAM 32 to 48 GB" src="https://img.shields.io/badge/RAM-32%E2%80%9348%20GB-FFA931?style=flat-square&labelColor=555">
  <img alt="Code GPL-3.0, weights Fish Audio Community License" src="https://img.shields.io/badge/license-GPL--3.0%20%C2%B7%20Fish%20Audio%20Community-FFA931?style=flat-square&labelColor=555">
</p>

<p align="center">
  <a href="https://pedromarinhodev.github.io/ComfyUI-Drama3/"><img alt="Listen to samples" src="assets/button-samples.svg" height="40"></a>
  <a href="#quick-start"><img alt="Quick start" src="assets/button-start.svg" height="40"></a>
  <a href="https://huggingface.co/PedroMarinhoDev/Drama-3-ComfyUI"><img alt="Weights on Hugging Face" src="assets/button-weights.svg" height="40"></a>
  <a href="example_workflows"><img alt="Workflows" src="assets/button-workflows.svg" height="40"></a>
</p>

Run **[Fish Audio Drama 3](https://huggingface.co/fishaudio/drama-3)**, a 35B text-to-speech model for voice cloning,
dialogue and inline cues such as `[sighs]` or `[whispers]`, **natively in ComfyUI on cards from 8 GB**. 90% of the model
is routed experts and each audio frame uses only 8 of the 256 in a layer, so the experts stay in system RAM and the
GPU keeps a cache of the ones in use. The pack also brings a native **Qwen3-ASR** that writes a voice clip's
transcript for you, and nodes to save a cloned voice and load it again. Built with Fish Audio.

<a href="https://pedromarinhodev.github.io/ComfyUI-Drama3/#samples"><picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/samples-dark.png">
  <img alt="Six samples: voice cloning, inline cues, two speakers, five speakers, singing and Portuguese" src="assets/samples-light.png" width="100%">
</picture></a>

<sub>Drama 3 FP8 · experts in RAM · temperature 1.0, top_p 0.9, top_k 30 · RTX 3090 and RTX 4090 · no real person
cloned: the clones copy voices the model made up, the rest are voices it invents per seed. Click to listen.</sub>

<table>
  <tr>
    <td width="33%"><img src="assets/icons/clone.svg" width="24" alt=""><br><b>Clone a voice from seconds of audio</b><br>A clip and its transcript are all it takes; save the voice and reuse it without the codec.</td>
    <td width="33%"><img src="assets/icons/dialogue.svg" width="24" alt=""><br><b>Up to 16 speakers in one text</b><br>Start each line with <code>&lt;|speaker:N|&gt;</code> and give each speaker a voice.</td>
    <td width="33%"><img src="assets/icons/cues.svg" width="24" alt=""><br><b>Cues where they should land</b><br><code>[laughs]</code>, <code>[whispers]</code>, <code>[sighs]</code> and plain-language directions inside the text (<a href="#tags-and-directions">how to write them</a>).</td>
  </tr>
  <tr>
    <td><img src="assets/icons/long.svg" width="24" alt=""><br><b>Long texts in one go</b><br>Read in chunks the way Fish Audio's server does, each chunk hearing the ones before it.</td>
    <td><img src="assets/icons/ram.svg" width="24" alt=""><br><b>Experts in RAM</b><br>About 6 GB of the model stays on the GPU; the 32 GB of experts stream from RAM through a VRAM cache.</td>
    <td><img src="assets/icons/asr.svg" width="24" alt=""><br><b>Transcripts by Qwen3-ASR</b><br>Native 1.7B and 0.6B speech recognition in 30 languages, no extra packages.</td>
  </tr>
</table>

## Quick start

1. **Install the nodes.** Clone into `custom_nodes` and restart ComfyUI. There's nothing to `pip install`: the nodes
   only use what ComfyUI already has.

   ```bash
   cd ComfyUI/custom_nodes
   git clone https://github.com/PedroMarinhoDev/ComfyUI-Drama3
   ```

2. **Download three files** from [PedroMarinhoDev/Drama-3-ComfyUI](https://huggingface.co/PedroMarinhoDev/Drama-3-ComfyUI)
   and [PedroMarinhoDev/Qwen3-ASR-ComfyUI](https://huggingface.co/PedroMarinhoDev/Qwen3-ASR-ComfyUI):

   | File | Folder | Size |
   |---|---|---|
   | `drama3_fp8experts_bf16.safetensors` | `models/diffusion_models` | 38.6 GB |
   | `drama3_codec.safetensors` | `models/vae` | 1.6 GB |
   | `qwen3_asr_1.7b_bf16.safetensors` | `models/asr` | 4.1 GB |

   ```bash
   hf download PedroMarinhoDev/Drama-3-ComfyUI drama3_fp8experts_bf16.safetensors --local-dir ComfyUI/models/diffusion_models
   hf download PedroMarinhoDev/Drama-3-ComfyUI drama3_codec.safetensors --local-dir ComfyUI/models/vae
   hf download PedroMarinhoDev/Qwen3-ASR-ComfyUI qwen3_asr_1.7b_bf16.safetensors --local-dir ComfyUI/models/asr
   ```

3. **Open the workflow and queue.** Drag [`example_workflows/drama3_voice_clone.json`](example_workflows/drama3_voice_clone.json)
   into ComfyUI, pick a few seconds of clean speech in **Load Audio**, write your text in **Drama 3 Generate**, run.

4. **Keep the voice.** Unmute **Save Drama 3 Voice** in the workflow: it writes the encoded voice (codes and transcript,
   a `.safetensors` of a few KB) to `output/drama3_voices`. **Load Drama 3 Voice** brings it back later, with no clip
   to load and nothing to encode or transcribe again.

> [!TIP]
> Qwen3-ASR fills in the voice's transcript. Cut the clip between sentences: a transcript that misses the clip's last
> words is the most common reason a take falls silent.

## Tags and directions

Directions are free natural language, not a fixed list: any wording works, and the examples below are Fish Audio's.
The full syntax is Appendix C of their [Drama 3 technical report](https://github.com/fishaudio/fish-speech/blob/main/docs/assets/Fish_Audio_Drama_3_Technical_Report.pdf).

| Control | How to write it | Examples |
|---|---|---|
| Speaker | `<\|speaker:N\|>` at the start of a line, N from 0 to 15; the voice for speaker N has the same tag at the start of its transcript (no tag means speaker 0) | `<\|speaker:0\|> Ready?` `<\|speaker:1\|> Always.` |
| Point tag | `[tag]` where the sound should land | `Well [sighs] fine, you win. [laughs]` `[pause]` `[gasps]` `[clears throat]` |
| Scoped tag | `<tag>...</tag>` around the words it shapes | `Promise me <whisper>you won't tell her</whisper>.` `<slow>...</slow>` `<emphasis>...</emphasis>` |
| Style prefix | `[style]` at the start of a line | `[with bright eager excitement] We won!` |
| Director's note | `Direction:` with the note, then `Line:` with the text on the next line | `Direction: icy calm, as if you already accepted it.` `Line: So this is goodbye.` |
| Language hint | `[language: xx]` after a line; it is not spoken | `Welcome back to the show. [language: en]` |

Text in round parentheses is spoken. Long text is cut into chunks of about 300 bytes without splitting a tag; keep a
director's note in one chunk with its line, or turn off `split_long_text`. Notes against an earlier take ("softer than
before") need that take in the prompt, which these nodes don't send.

## Requirements

| | Minimum | Tested |
|---|---|---|
| GPU | NVIDIA, 8 GB of VRAM with experts in RAM | RTX 4090 and RTX 3090 (24 GB); 12 GB and 8 GB simulated on the 3090 |
| System RAM | the model file plus ~8 GB: ~32 GB for W4A8, ~48 GB for FP8 | 188 GB |
| CPU | any; decoding speed follows its single-core speed | Intel Core i9-14900K |
| Disk | 26 GB (W4A8 and the codec) | NVMe |
| ComfyUI | v0.40 | v0.40.0 |

## Choosing a file

**Start with FP8**: it holds Fish Audio's own expert weights, byte for byte, next to the checkpoint's bf16 layers.
The int8 file is the same size, measured as close to the original, and ran 14–17% faster than FP8 on an RTX 3090:
the nodes turn FP8 experts into bf16 each time they run, while int8 experts run as int8. Take a smaller file when your
RAM is short of the FP8 file's ~48 GB.

| File | Size | Experts | RAM with experts in RAM | Notes |
|---|---|---|---|---|
| `drama3_fp8experts_bf16` | 38.6 GB | FP8, as Fish Audio ships them | ~48 GB | the original weights |
| `drama3_int8experts_bf16` | 38.7 GB | int8 ConvRot | ~48 GB | faster than FP8 on the 3090; no silent take in 12 cloned test takes |
| `drama3_w6a8experts_bf16` | 32.7 GB | W6A8 | ~40 GB | no silent take in 16 cloned test takes |
| `drama3_w4a8experts_bf16` | 24.6 GB | W4A8 | ~32 GB | falls silent more often with real voice clips |

Qwen3-ASR comes in two sizes, `qwen3_asr_1.7b_bf16` (4.1 GB) and `qwen3_asr_0.6b_bf16` (1.6 GB). In our tests both
wrote nearly the same transcripts of clean speech, and the 0.6B took half the time.

## Performance

Drama 3 makes 21.5 audio frames per second of speech, so **46 ms per frame is real time**. Experts in RAM, a cloned
voice, ~9 s of speech, measured on 2026-10-15 with an Intel Core i9-14900K (the held rows on 2026-10-14); the time
includes the prompt pass. One frame at a time, decoding waits on the CPU (one core launching many small GPU kernels):
int8 runs at the same 69 ms per frame on the RTX 3090 and the RTX 4090, and a CPU with slower cores adds to these times.

| GPU | File | Expert cache | ms per frame | Peak VRAM |
|---|---|---|---|---|
| RTX 4090 (PCIe 4.0 ×8) | W4A8 | 16 GB | 52 | 21.3 GB |
| RTX 4090 (PCIe 4.0 ×8) | W4A8 | 4 GB | 67 | 11.7 GB |
| RTX 4090 (PCIe 4.0 ×8) | int8 | 16 GB | 69 | 21.5 GB |
| RTX 4090 (PCIe 4.0 ×8) | int8 | 4 GB | 103 | 11.9 GB |
| RTX 3090 (PCIe 4.0 ×8) | int8 | 16 GB | 69 | 21.6 GB |
| RTX 3090 (PCIe 4.0 ×8) | int8 | 4 GB | 102 | 11.9 GB |
| RTX 3090 (PCIe 4.0 ×8) | FP8 | 16 GB | 84 | 21.6 GB |
| RTX 3090 (PCIe 4.0 ×8) | FP8 | 4 GB | 119 | 11.9 GB |
| RTX 3090 held to 12 GB | FP8 | automatic | 137 | 9.8 GB |
| RTX 3090 held to 8 GB | FP8 | automatic | 214 | 6.5 GB |

The held rows ran with the rest of the 3090's memory taken by another process, as on a 12 GB or 8 GB card. Qwen3-ASR transcribes a 6.6 s clip in ~2 s with the 1.7B model and ~1 s with the 0.6B (RTX 3090,
model load included).

<details>
<summary><b>Nodes</b></summary>

| Node | What it does |
|---|---|
| **Load Drama 3** | Loads a model file. `experts`: in RAM (default) or in VRAM; `expert_cache_gb`: VRAM for cached experts (0 = what is free); advanced: a second GPU for the experts-in-VRAM split. |
| **Load Drama 3 Codec** | Loads the codec from `models/vae`. |
| **Drama 3 Encode Voice** | A voice to clone: an audio clip and its exact transcript. |
| **Save Drama 3 Voice** / **Load Drama 3 Voice** | Keep an encoded voice (codes and transcript, a few KB) in `output/drama3_voices` and use it again without the codec. |
| **Drama 3 Generate** | Text and any number of voices to audio codes. Long text is generated in chunks, each continuing from the first and the latest chunk. |
| **Drama 3 Decode** | Codes to 44.1 kHz audio. |
| **Load Qwen3-ASR** | Loads a Qwen3-ASR file from `models/asr`. |
| **Qwen3-ASR Transcribe** | Audio to text and its language; `language` detects it or forces one of 30; `context` takes names and terms that help recognition. |

The codec runs only before and after generation, so it never sits in VRAM next to the model.

</details>

<details>
<summary><b>How it fits on an 8 GB card</b></summary>

Drama 3 is a Qwen3.5-style mixture of experts: 40 layers, each with 256 routed experts of which 8 run per frame,
plus attention or DeltaNet, a shared expert and a 13-layer codebook decoder. Everything except the routed experts is
about 6 GB and stays on the GPU. With **experts in RAM** the 32 GB of experts stay in pinned system memory: each frame
copies the 8 experts per layer it needs, and a least-recently-used cache in the VRAM left free keeps the ones that
come back (85–95% of reads hit a 13 GB cache). The prompt uses nearly every expert, so its pass streams each layer's
experts once and keeps the ones the prompt used most. Decoding one frame at a time is CPU-bound, so the cache size
moves the speed less than you would expect.

With **experts in VRAM** the whole file sits on the GPU; two 24 GB cards can split the layers (`second_gpu`).

</details>

<details>
<summary><b>Settings</b></summary>

- **temperature 1.0, top_p 0.9, top_k 30** are the defaults of Fish Audio's own Drama 3 server. Lower values make
  takes slower, flatter and more likely to fall silent.
- The model copies the reference's delivery: a flat clip gives a flat take, a lively one a lively take.
- **max_frames** caps each chunk (21.5 frames per second); generation stops earlier at the end of speech.

</details>

<details>
<summary><b>Building the files yourself</b></summary>

From the official checkpoints, one GPU visible (`CUDA_VISIBLE_DEVICES=N`), with ComfyUI on `PYTHONPATH`:

```bash
python tools/convert.py model <drama-3 folder> drama3_fp8experts_bf16.safetensors --format fp8
python tools/convert.py model <drama-3 folder> drama3_int8experts_bf16.safetensors --format int8_convrot
python tools/convert.py model <drama-3 folder> drama3_w6a8experts_bf16.safetensors --format w6a8
python tools/convert.py model <drama-3 folder> drama3_w4a8experts_bf16.safetensors --format w4a8
python tools/convert.py codec <drama-3 folder>/codec.pth drama3_codec.safetensors
python tools/convert.py asr <Qwen3-ASR-1.7B folder> qwen3_asr_1.7b_bf16.safetensors
python tools/convert.py asr <Qwen3-ASR-0.6B folder> qwen3_asr_0.6b_bf16.safetensors
```

</details>

<details>
<summary><b>How close it is to the official code</b></summary>

- **Drama 3**: against Fish Audio's own PyTorch code, with the same frames forced, the main token's distribution
  differs by a KL of 0.004–0.007 nats and the top token agrees 93–94% of the time, the noise of bf16 arithmetic;
  the FP8 and int8 files measure the same. Prompt rows match the builders of Fish Audio's server and of fish-speech
  token for token, and the codec produces 99.94% of the same codes.
- **Qwen3-ASR**: against Qwen's `qwen-asr` package, the prompt ids are identical, the encoder output matches within
  1e-6 in fp32 and the first token is the same; transcripts match on clips up to 8 s. On longer clips the encoder
  attends in Qwen's 8 s windows, as their vLLM backend does, which can move a comma.

</details>

## Troubleshooting

<details>
<summary><b>A take falls silent after a second, or speaks only part of the text</b></summary>

Check the voice first: the transcript must cover every word of the clip, and the clip should be clean speech cut
between sentences. Then queue again for a new seed. Some clips make the model fall silent in Fish Audio's own code
too (known issue); try another stretch of the recording. A take that goes silent is stopped and its silent tail cut,
with a warning in the console.

</details>

<details>
<summary><b>Speech is slow or flat</b></summary>

Keep temperature at 1.0 and top_p at 0.9 or above, and use a reference with the delivery you want.

</details>

<details>
<summary><b>Out of memory, or the first run takes minutes</b></summary>

Lower `expert_cache_gb`, or pick a smaller file if system RAM is the limit. The first run reads the whole model file
from disk once; later runs start in seconds.

</details>

## License and credits

The code in this repository is under the [GNU GPL v3](LICENSE), except the parts ported from Fish Audio's code and
Fish Audio's tokenizer and config, which remain under the [Fish Audio Community License](LICENSE-FISH-AUDIO), and the
parts adapted from Qwen's `qwen-asr`, under the [Apache License 2.0](LICENSE-APACHE). [NOTICE](NOTICE) lists them.

> [!NOTE]
> Drama 3's weights and outputs fall under the Fish Audio Community License: free for research and non-commercial
> use, and for commercial use below USD 500k annual revenue; a hosted TTS service needs a license from Fish Audio.
> Don't clone the voice of an identifiable person without their consent.

**Built with Fish Audio.** Drama 3 and its codec are by [Fish Audio](https://fish.audio); Qwen3-ASR is by the
[Qwen team](https://github.com/QwenLM/Qwen3-ASR). This project is not affiliated with or endorsed by either.
