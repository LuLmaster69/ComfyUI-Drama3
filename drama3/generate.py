"""Autoregressive frame generation: prefill the prompt rows, then per frame sample the main token from the audio
head, the other codes from the tower, and run the backbone on the new frame."""
import contextlib
import logging

import torch

import comfy.model_management
import comfy.ops
from .config import MIN_TEMPERATURE, RAS_TEMPERATURE, RAS_TOP_K, RAS_TOP_P, RAS_WINDOW

# A take that falls silent without ending repeats a handful of codes: 96 frames (4.5 s) of speech hold 80-95
# distinct first codes and the most expressive measured stretch (sighs, laughs) 41; silent runaways 5-20.
STALL_WINDOW = 96
STALL_DISTINCT = 24
STALL_TAIL = 10


def sample_top_k_top_p(logits, temperature, top_k, top_p, generator):
    """One id per row; drops the token crossing top_p except the highest-ranked one (the vendor's rule)."""
    values, ids = torch.sort(logits.float() / max(temperature, MIN_TEMPERATURE), descending=True, dim=-1)
    ranks = torch.arange(values.shape[-1], device=values.device)
    dropped = (values.softmax(-1).cumsum(-1) > top_p) | (ranks >= top_k)
    dropped[..., 0] = False
    probabilities = values.masked_fill(dropped, -torch.inf).softmax(-1)
    noise = torch.empty_like(probabilities).exponential_(generator=generator)
    return ids.gather(-1, (probabilities / noise).argmax(-1, keepdim=True)).squeeze(-1)


def embed_rows(embed, ids, dtype):
    """Token embeddings; an evicted dynamic table gathers the rows from its host copy instead of streaming whole."""
    if (hasattr(embed, "_v") and embed.weight_lowvram_function is None and len(embed.weight_function) == 0
            and comfy.ops.vbar_above_watermark(embed)):
        return embed.host_rows(ids, out_dtype=dtype)
    return embed(ids, out_dtype=dtype)


def generate(drama3, prompt, max_frames, temperature, top_k, top_p, seed, on_frame=None):
    """Frames [frames, 1 + N] (CPU int64) sampled after the prompt rows [rows, 1 + N], without the end frame."""
    config = drama3.config
    backbone = drama3.model.model
    dtype = drama3.model.dtype
    layers, devices, norm, audio_head, tower = drama3.parts()
    first, last = devices[0], devices[-1]
    rows = prompt.to(last)
    prompt_rows = len(rows)
    capacity = prompt_rows + max_frames + 1
    caches = backbone.init_caches(capacity, devices, dtype)
    tower_caches = tower.init_caches(capacity, last, dtype)
    generator = torch.Generator(device=last).manual_seed(seed)
    candidates = torch.cat((torch.tensor([config.eos_token_id], device=last),
                            torch.arange(config.semantic_start_token_id, config.semantic_end_token_id + 1, device=last)))

    def run(frames):
        text = embed_rows(backbone.embed_tokens, frames[:, 0].to(first), dtype)
        x = backbone.run_layers(tower.embed_backbone(frames, text)[None], caches, layers, devices)
        with comfy.model_management.cuda_device_context(last):
            return norm(x)

    def sample(logits):
        return sample_top_k_top_p(logits, temperature, top_k, top_p, generator)

    frames, codes = [], []
    session = drama3.experts.session(first, drama3.patchers()) if drama3.experts is not None else contextlib.nullcontext()
    with session:
        hidden = run(rows)
        history = torch.full((RAS_WINDOW,), -1, dtype=torch.long, device=last)
        window = rows[-RAS_WINDOW:, 0]
        history[RAS_WINDOW - len(window):] = window
        frame = rows[-1:]
        with comfy.model_management.cuda_device_context(last):
            if prompt_rows > 1:
                count = prompt_rows - 1
                tower.run(hidden[:, :-1], rows[1:], rows[:-1], tower.context(torch.arange(count, device=last), count, last, dtype), tower_caches)
            hidden = hidden[:, -1:]
            for step in range(max_frames):
                position = prompt_rows - 1 + step
                logits = audio_head(hidden[0])
                draw = candidates[sample(logits)]
                retry = candidates[sample_top_k_top_p(logits, RAS_TEMPERATURE, RAS_TOP_K, RAS_TOP_P, generator)]
                repeated = (history == draw[:, None]).any(-1) & tower.is_semantic(draw)
                main = torch.where(repeated, retry, draw)
                frame = tower.step(main, hidden, frame, tower.context(torch.tensor([position], device=last), 0, last, dtype), tower_caches, sample)
                main_id, code, finite = torch.stack((frame[0, 0], frame[0, 1], torch.isfinite(logits).all().long())).tolist()
                if not finite:
                    raise FloatingPointError("Drama 3 produced non-finite logits")
                if main_id == config.eos_token_id:
                    break
                frames.append(frame)
                codes.append(code)
                if len(codes) >= STALL_WINDOW and len(set(codes[-STALL_WINDOW:])) <= STALL_DISTINCT:
                    logging.warning("Drama 3: the take went silent without ending after %.1f s; its silent tail was cut. "
                                    "Another seed or a voice reference usually fixes this.", len(frames) / 21.5)
                    del frames[len(frames) - STALL_WINDOW + STALL_TAIL:]
                    break
                history = torch.cat((history[1:], frame[:, 0]))
                if on_frame is not None:
                    on_frame(step)
                if step + 1 < max_frames:
                    hidden = run(frame)
    if not frames:
        return torch.zeros((0, rows.shape[1]), dtype=torch.long)
    return torch.cat(frames).cpu()
