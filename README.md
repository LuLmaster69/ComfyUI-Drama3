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

> [!NOTE]
> **3× faster since 2026-10-17.** The pack's own CUDA kernels read the experts the cache misses straight from RAM while
> the CPU computes others, and every layer replays as a CUDA graph: one RTX 3090 now speaks a 9 s line in 6.7 s instead
> of 16.7 s, about twice as fast as real time, two GPUs take 2.8 s, and an 8 GB card keeps pace with real time.
> Nothing new to install. [Numbers](#performance).

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
    <td><img src="assets/icons/ram.svg" width="24" alt=""><br><b>Faster than real time, experts in RAM</b><br>About 6 GB of the model stays on the GPU; custom CUDA kernels read the 32 GB of experts from RAM through a VRAM cache.</td>
    <td><img src="assets/icons/asr.svg" width="24" alt=""><br><b>Transcripts by Qwen3-ASR</b><br>Native 1.7B and 0.6B speech recognition in 30 languages, no extra packages.</td>
  </tr>
</table>

## Quick start

1. **Install the nodes** and restart ComfyUI. Search for **Drama 3 text to speech** in ComfyUI-Manager, or install it
   from the [Comfy Registry](https://registry.comfy.org/nodes/drama3) with `comfy node install drama3`. Or clone it
   into `custom_nodes`. There's nothing to `pip install`: the nodes only use what ComfyUI already has.

   ```bash
   cd ComfyUI/custom_nodes
   git clone https://github.com/PedroMarinhoDev/ComfyUI-Drama3
   ```

2. **Download the files** from [PedroMarinhoDev/Drama-3-ComfyUI](https://huggingface.co/PedroMarinhoDev/Drama-3-ComfyUI)
   and [PedroMarinhoDev/Qwen3-ASR-ComfyUI](https://huggingface.co/PedroMarinhoDev/Qwen3-ASR-ComfyUI):

   | File | Folder | Size |
   |---|---|---|
   | `drama3_fp8experts_bf16.safetensors` | `models/diffusion_models` | 38.6 GB |
   | `drama3_codec.safetensors` | `models/vae` | 1.6 GB |
   | `qwen3_asr_1.7b_bf16.safetensors` | `models/asr` | 4.1 GB |
   | `qwen3_forced_aligner_0.6b_bf16.safetensors` (optional: word timing in Drama 3 Editor) | `models/asr` | 1.8 GB |

   ```bash
   hf download PedroMarinhoDev/Drama-3-ComfyUI drama3_fp8experts_bf16.safetensors --local-dir ComfyUI/models/diffusion_models
   hf download PedroMarinhoDev/Drama-3-ComfyUI drama3_codec.safetensors --local-dir ComfyUI/models/vae
   hf download PedroMarinhoDev/Qwen3-ASR-ComfyUI qwen3_asr_1.7b_bf16.safetensors --local-dir ComfyUI/models/asr
   hf download PedroMarinhoDev/Qwen3-ASR-ComfyUI qwen3_forced_aligner_0.6b_bf16.safetensors --local-dir ComfyUI/models/asr
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
| Speaker | `<\|speaker:N\|>` at the start of a line, N from 0 to 15; the voice in Generate's slot `voiceN` speaks for speaker N, unless its transcript starts with a tag of its own. The lines reach the model joined by `<\|audio_separator\|>`, the dialogue layout of Fish's report (Figure 11) | `<\|speaker:0\|> Ready?` `<\|speaker:1\|> Always.` |
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
| CPU | any; with int8 experts in RAM it computes part of the experts, so more cores help on small cards | Intel Core i9-14900K |
| Disk | 26 GB (W4A8 and the codec) | NVMe |
| ComfyUI | v0.40 | v0.41.0 and master after v0.41.2 |

## Choosing a file

**Start with FP8**: it holds Fish Audio's own expert weights, byte for byte, next to the checkpoint's bf16 layers.
**On a card under 24 GB take int8**: it is the same size and measured as close to the original, and only int8 experts
are computed by the CPU when the VRAM cache misses them, which makes it twice as fast as FP8 with a 4 GB cache. Take a
smaller file when your RAM is short of the FP8 file's ~48 GB; on a 24 GB card they are also the fastest.

| File | Size | Experts | RAM with experts in RAM | Notes |
|---|---|---|---|---|
| `drama3_fp8experts_bf16` | 38.6 GB | FP8, as Fish Audio ships them | ~48 GB | the original weights |
| `drama3_int8experts_bf16` | 38.7 GB | int8 ConvRot | ~48 GB | the fastest on cards under 24 GB; no silent take in 12 cloned test takes |
| `drama3_w6a8experts_bf16` | 32.7 GB | W6A8 | ~40 GB | no silent take in 16 cloned test takes |
| `drama3_w4a8experts_bf16` | 24.6 GB | W4A8 | ~32 GB | falls silent more often with real voice clips |

Qwen3-ASR comes in two sizes, `qwen3_asr_1.7b_bf16` (4.1 GB) and `qwen3_asr_0.6b_bf16` (1.6 GB). In our tests both
wrote nearly the same transcripts of clean speech, and the 0.6B took half the time. `qwen3_forced_aligner_0.6b_bf16`
(1.8 GB, also on [Qwen3-ASR-ComfyUI](https://huggingface.co/PedroMarinhoDev/Qwen3-ASR-ComfyUI)) gives Drama 3 Editor the time of every word; on a 40 s
take its times match Qwen's own aligner within 0.16 s, 121 of 131 words exactly.

## Split VRAM + RAM

The experts-in-RAM mode keeps every routed expert in RAM and only copies the most used into the VRAM cache, so it
needs the whole file in RAM: the int8 file does not run on 24 GB of VRAM and 32 GB of RAM. When RAM is the limit, pick
**split VRAM + RAM** in `experts` on **Load Drama 3** and leave `vram_expert_layers` at 0. The first layers' experts
then stay in VRAM for good, loaded straight from the file, and only the other layers' go to RAM, behind the same cache
and kernels. The loader counts the VRAM it can have (what is free once ComfyUI has unloaded its other models) and
keeps there the rest of the model (about 6 GB) and the experts of as many layers as fit (0.75 GiB each with int8 or
FP8, 0.43 with W4A8), with 3 GiB left for the expert cache, a prompt's activations and the decode graphs. The other
layers' experts go to RAM, *pinned* (locked for the GPU to read) as far as the system allows: Windows pins at most
about half the RAM for all of ComfyUI, so the experts take at most half less 2 GiB and less what ComfyUI already
pinned, about 14 GiB of 32 GB. With int8, experts past that stay in pageable RAM and the CPU runs the ones the cache
misses; with the other formats the loader keeps one or two more layers in VRAM to pin them all, and failing that,
generation takes a slow path. The console says what it chose:

```
Drama 3: experts of layers 0-17 in VRAM on cuda:0 (13.6 GiB, the rest of the model 5.9 GiB), of layers 18-39 in RAM (16.6 GiB)
Drama 3: 14.0 GiB of experts are pinned in RAM and 2.6 GiB stay pageable (...): the CPU runs the experts of those layers the cache misses
```

| File | 24 GB card, 32 GB of RAM, Windows |
|---|---|
| W4A8 | 32 layers in VRAM, 3.4 GiB in RAM. Measured on an RTX 3090: 94% of expert reads hit the cache, and a 41 s take generated in about 30 s |
| W6A8 | about 22 layers in VRAM, 11 GiB in RAM: all pinned while ComfyUI itself holds under 3 GiB pinned |
| int8 | 18 layers in VRAM, 16.6 GiB in RAM: up to 14 GiB pinned, the CPU runs the misses of the rest (not yet run end to end) |
| FP8 | 18 layers in VRAM, 16.6 GiB in RAM: more than Windows pins, so the slow path. Take int8 instead |

On Linux the experts may take the RAM available less 4 GiB, so with 32 GB the int8 file's 16.6 GiB are all pinned.
Only the W4A8 row has been timed; the others are the loader's own choices.

- **Fewer layers in VRAM** (`vram_expert_layers` 12 or 16, say) leave more VRAM to the cache, which holds the most used
  experts of every layer and so serves more reads per GB than whole layers do. That is faster, but every layer you take
  out of VRAM puts 0.75 GiB more in RAM. On Linux with RAM to spare this is worth trying.
- **More layers than fit** make ComfyUI stream what does not fit from RAM at every frame, which is slow; the console
  warns.
- **VRAM held outside ComfyUI's models** cannot be counted on: another program, or a custom node that keeps its own
  model on the GPU (a `transformers` speech recognition node, for one). The loader splits for the VRAM left once
  ComfyUI has unloaded its own models and says how much was held; a generation that finds too little VRAM free says so
  too. This pack's Qwen3-ASR nodes are ComfyUI models, unloaded when Drama 3 needs the room, and a voice saved with
  **Save Drama 3 Voice** needs no recognition at all.
- **Experts that cannot be pinned** stay in pageable RAM (the model file's own pages), where the GPU cannot read them.
  With int8 the CPU runs those layers' missed experts and the kernels stay on. With the other formats generation runs
  without the kernels, a few times slower, with a warning naming the GiB involved: keep more layers in VRAM, close what
  else holds RAM, or take the int8 file. Loading Drama 3 again (another `vram_expert_layers`) while the earlier load is
  still alive pins twice: restart ComfyUI, or free its models and node cache first.
- Split experts run on one GPU: `second_gpu` must stay at none.
- The split needs ComfyUI's dynamic VRAM, on by default on NVIDIA cards (the console says "DynamicVRAM support detected
  and enabled"). Without it, ComfyUI first loads the whole file into RAM, which 32 GB cannot hold.

## Performance

Drama 3 makes 21.5 audio frames per second of speech, so **46 ms per frame is real time**. Since 2026-10-17 one 24 GB
card decodes 2–4× faster than real time, and a 9 s line takes 4–8 s in all on one GPU, under 3 s on two. Measured on
2026-10-17 as ComfyUI runs the nodes (dynamic VRAM), a cloned voice, a 9 s line, the mean of two seeds (five for int8
with a 4 GB cache on the 3090), with an Intel Core i9-14900K and ComfyUI v0.41; *before* is the pack of 2026-10-16
measured the same way. The line's time includes the prompt pass.

| GPU | File | Expert cache | ms per frame, before → now | 9 s line, before → now | Peak VRAM |
|---|---|---|---|---|---|
| RTX 4090 (PCIe 4.0 ×8) | W4A8 | 16 GB | 56.9 → **11.7** | 13.6 → **3.9 s** | 21.8 GB |
| RTX 4090 (PCIe 4.0 ×8) | int8 | 16 GB | 67.0 → **20.2** | 16.4 → **6.1 s** | 21.8 GB |
| RTX 4090 (PCIe 4.0 ×8) | W4A8 | 4 GB | 75.3 → **29.5** | 16.9 → **6.9 s** | 10.7 GB |
| RTX 4090 (PCIe 4.0 ×8) | int8 | 4 GB | 102.9 → **32.7** | 23.0 → **7.6 s** | 10.8 GB |
| RTX 3090 (PCIe 4.0 ×8) | W4A8 | 16 GB | 56.7 → **14.4** | 13.3 → **4.7 s** | 21.8 GB |
| RTX 3090 (PCIe 4.0 ×8) | W6A8 | 16 GB | 61.5 → **19.2** | 14.7 → **6.2 s** | 21.8 GB |
| RTX 3090 (PCIe 4.0 ×8) | int8 | 16 GB | 68.8 → **23.2** | 16.7 → **6.7 s** | 21.8 GB |
| RTX 3090 (PCIe 4.0 ×8) | FP8 | 16 GB | 67.6 → **24.8** | 16.6 → **7.7 s** | 21.8 GB |
| RTX 3090 (PCIe 4.0 ×8) | int8 | 4 GB | 104.2 → **34.4** | 22.9 → **8.0 s** | 10.6 GB |
| RTX 3090 (PCIe 4.0 ×8) | W4A8 | 4 GB | 76.2 → **31.7** | 16.7 → **7.7 s** | 10.6 GB |
| RTX 3090 (PCIe 4.0 ×8) | FP8 | 4 GB | 107.5 → **58.0** | 23.4 → **13.5 s** | 10.6 GB |
| RTX 3090 held to 12 GB | int8 | automatic | 110.1 → **37.5** | 24.0 → **8.4 s** | 9.6 GB |
| RTX 3090 held to 12 GB | FP8 | automatic | 115.1 → **63.9** | 24.8 → **14.5 s** | 9.6 GB |
| RTX 3090 held to 8 GB | int8 | automatic | 145.2 → **45.8** | 30.6 → **9.9 s** | 6.7 GB |
| RTX 3090 held to 8 GB | FP8 | automatic | 148.6 → **97.3** | 30.7 → **20.7 s** | 6.7 GB |
| RTX 3090 + RTX 4090, experts in VRAM | int8 | – | 56.7 → **13.2** | 12.6 → **2.8 s** | 18.8 + 18.7 GB |
| RTX 3090 + RTX 4090, experts in VRAM | FP8 | – | 48.9 → **12.9** | 10.7 → **2.9 s** | 18.7 + 18.5 GB |

The held rows ran with the rest of the 3090's memory taken by another process, as on a 12 GB or 8 GB card. With int8
experts the CPU computes part of the experts the cache misses, so int8 holds up on small cards where FP8 slows down: a
CPU with fewer or slower cores gets less of that help. ComfyUI after v0.41.2 decodes about 2% faster again. Qwen3-ASR
transcribes a 6.6 s clip in ~2 s with the 1.7B model and ~1 s with the 0.6B (RTX 3090, model load included).

<details>
<summary><b>Nodes</b></summary>

| Node | What it does |
|---|---|
| **Load Drama 3** | Loads a model file. `experts`: in RAM (default), in VRAM, or split VRAM + RAM; `vram_expert_layers`: with the split, how many layers keep their experts in VRAM (0 = as many as fit); `expert_cache_gb`: VRAM for cached experts (0 = what is free); advanced: a second GPU for the experts-in-VRAM split. |
| **Load Drama 3 Codec** | Loads the codec from `models/vae`. |
| **Drama 3 Encode Voice** | A voice to clone: an audio clip and its exact transcript. |
| **Save Drama 3 Voice** / **Load Drama 3 Voice** | Keep an encoded voice (codes and transcript, a few KB) in `output/drama3_voices` and use it again without the codec. |
| **Drama 3 Generate** | Text and any number of voices to audio codes. Long text is generated in chunks, each continuing from the first and the latest chunk; `speakers` can generate each speaker on their own (per line or per speaker). |
| **Save Drama 3 Codes** / **Load Drama 3 Codes** | Keep a generated take (codes and line timing, a few hundred KB) in `output/drama3_codes` and decode or edit it again without generating it again. |
| **Drama 3 Decode** | Codes to 44.1 kHz audio; separately generated takes are joined with 0.3 s of silence. |
| **Drama 3 Editor** | Codes to audio, with captions that follow the voice word by word (with Qwen3-ForcedAligner on `aligner`) and one track per voice. *Edit* opens the editor over ComfyUI: pick how the silences sound (as generated, natural, tight, or your own values), check the lines it flags (a line short or long for its words, long silences), cut, add silence, name the voices and set their volume. *Use these edits* runs the node again with them; they stay in the workflow. |
| **Load Qwen3-ASR** | Loads a Qwen3-ASR or Qwen3-ForcedAligner file from `models/asr`. |
| **Qwen3-ASR Transcribe** | Audio to text and its language; `language` detects it or forces one of 30; `context` takes names and terms that help recognition. |

The codec runs only before and after generation, so it never sits in VRAM next to the model.

</details>

<details>
<summary><b>How it fits on an 8 GB card</b></summary>

Drama 3 is a Qwen3.5-style mixture of experts: 40 layers, each with 256 routed experts of which 8 run per frame,
plus attention or DeltaNet, a shared expert and a 13-layer codebook decoder. Everything except the routed experts is
about 6 GB and stays on the GPU. With **experts in RAM** the 32 GB of experts stay in pinned system memory and the VRAM
left free caches the most used ones (83% of reads hit a 15 GB cache); the cache lasts a whole Generate run.

The pack's CUDA kernels, compiled at run time by PyTorch's NVRTC (no compiler or package to install), look each expert up
in the cache on the GPU, run the cached ones in place and read the missed ones straight from RAM over PCIe, filling
free cache slots as they go. With int8 experts the CPU computes part of the misses at the same time, and the most used
of them are copied into the cache between frames. Since nothing in a frame waits on the CPU, every layer and the
codebook decoder replay as CUDA graphs. The prompt pass groups its rows by expert and splits the experts between the
GPU and the CPU the same way. Where PyTorch can't compile the kernels the nodes copy the experts instead, at the
previous version's speed.

With **experts in VRAM** the whole file sits on the GPU and the same kernels read the experts in place; two 24 GB
cards can split the layers (`second_gpu`).

</details>

<details>
<summary><b>Settings</b></summary>

- **temperature 1.0, top_p 0.9, top_k 0** (no cap) are the defaults of Fish Audio's Drama 3 Space. Their server's
  top_k 30 makes takes fall silent at pauses: 11 of 40 takes in Fish Audio's test, 2 of 4 cloned takes here, each
  losing a sentence. Lower temperature or top_p make takes slower, flatter and more likely to fall silent.
- The model copies the reference's delivery: a flat clip gives a flat take, a lively one a lively take.
- **max_frames** caps each chunk (21.5 frames per second); generation stops earlier at the end of speech.
- **speakers** sets how a dialogue is generated. *one take per line* (default) generates each line on its own, with
  only its speaker's voices and earlier takes in the prompt, as Fish Audio's Space voices a scene: voices cannot bleed
  into each other, but lines do not hear each other. *one take per line, hearing the line before* adds the take before
  as context, so lines react again but that voice can bleed in. *one take per speaker* says all of a speaker's lines in
  one take (in 300-byte chunks with `split_long_text`), cuts it at the line breaks it says and puts the lines back in
  script order. *together* is one script in which every line hears the others. Decode joins separate takes with 0.3 s
  of silence. Pause cues are riskier in a solo take: a line that opens with `[long pause]` can stay silent.
- As in Fish Audio's Space: **Encode Voice** trims the clip's silent edges and normalizes its peak; **Generate** takes a
  chunk once more when it has no sound or stalls in its first 2.5 s; **Decode**'s `shorten_long_pauses` (on) shortens
  pauses over 0.75 s to 0.75 s, cuts the silence before the first sound and keeps 0.15 s at the end.

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
python tools/convert.py asr <Qwen3-ForcedAligner-0.6B folder> qwen3_forced_aligner_0.6b_bf16.safetensors
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

Set `top_k` to 0: workflows saved before this fix keep the old 30, which makes takes fall silent at pauses. Then check
the voice: the transcript must cover every word of the clip, and the clip should be clean speech cut between
sentences. Some clips make the model fall silent in Fish Audio's own code too (known issue); try another stretch of the
recording. A take that goes silent is stopped and its silent tail cut, with a warning in the console; a new seed
usually helps.

</details>

<details>
<summary><b>Speakers swap voices in a dialogue</b></summary>

Give every speaker a voice. A long dialogue is generated in chunks, and an earlier chunk's take in the prompt hands the
voice it ends in to the next chunk: with two cloned voices, 6 of 16 lines after the first chunk came out in the other
voice. When every speaker has a voice, Generate leaves earlier takes out of dialogue chunks and the references keep
the voices apart (1 of 18). A speaker without a voice still needs them to keep the voice the model made up.

</details>

<details>
<summary><b>Speech is slow or flat</b></summary>

Keep temperature at 1.0 and top_p at 0.9 or above, and use a reference with the delivery you want.

</details>

<details>
<summary><b>Out of memory, or the first run takes minutes</b></summary>

Lower `expert_cache_gb`, or pick a smaller file if system RAM is the limit. When RAM is short but VRAM is not, use
**split VRAM + RAM** ([above](#split-vram--ram)), which keeps part of the experts in VRAM instead of RAM. The first run reads the whole model file
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
