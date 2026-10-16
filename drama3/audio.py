"""Clean-up around Drama 3 as Fish Audio's Drama 3 Space does it (server.py `tidy_reference` and `SilenceGuard`).

A voice clip loses the silence at its edges and is peak-normalized before it is encoded. A take loses the silence
before its first sound, keeps every pause up to a limit, and ends with a short tail of its silence.
"""
import torch
import torch.nn.functional as F

# A 50 ms window quieter than this RMS is silence.
SILENCE_RMS = 0.003
SILENCE_WINDOW = 0.05
TAIL = 0.15
# Edge silence of a clip: frames 40 dB below its loudest (librosa.effects.trim with 1024 / 256 frames).
CLIP_TOP_DB = 40
CLIP_FRAME = 1024
CLIP_HOP = 256
CLIP_PEAK = 0.9
CLIP_MAX_GAIN = 10.0


def tidy_clip(waveform):
    """A mono clip [1, samples] without its silent edges, peak-normalized to CLIP_PEAK (at most CLIP_MAX_GAIN)."""
    waveform = torch.nan_to_num(waveform)
    padded = F.pad(waveform, (CLIP_FRAME // 2, CLIP_FRAME // 2))
    rms = padded.unfold(-1, CLIP_FRAME, CLIP_HOP).pow(2).mean(-1).sqrt()[0]
    loud = (rms.clamp(min=1e-5) > rms.max().clamp(min=1e-5) * 10 ** (-CLIP_TOP_DB / 20)).nonzero()[:, 0].tolist()
    if loud:
        waveform = waveform[:, loud[0] * CLIP_HOP:min(waveform.shape[-1], (loud[-1] + 1) * CLIP_HOP)]
    peak = waveform.abs().max().item()
    if peak > 0:
        waveform = waveform * min(CLIP_MAX_GAIN, CLIP_PEAK / peak)
    return waveform.clamp(-1.0, 1.0)


def trim_pauses(waveform, sample_rate, max_pause):
    """A take [channels, samples] without the silence before its first sound, each pause cut to its first
    `max_pause` seconds and the end to TAIL seconds. A take without any sound is returned as it is."""
    window = max(1, int(SILENCE_WINDOW * sample_rate))
    rms = torch.stack([piece.mean() for piece in waveform.pow(2).mean(0).split(window)]).sqrt()
    quiet = (rms < SILENCE_RMS).tolist()
    if all(quiet):
        return waveform
    pieces, held, started = [], 0, False
    for index, silent in enumerate(quiet):
        start = index * window
        if not silent:
            started = True
            held = 0
            pieces.append((start, min(start + window, waveform.shape[-1])))
        elif started:
            keep = min(window, max(0, int(max_pause * sample_rate) - held))
            if keep:
                pieces.append((start, start + keep))
            held += keep
    # the end keeps at most TAIL of the silence after the last sound
    last = max(index for index, silent in enumerate(quiet) if not silent)
    limit = (last + 1) * window + int(TAIL * sample_rate)
    kept = [(start, min(end, limit)) for start, end in pieces if start < limit]
    return torch.cat([waveform[:, start:end] for start, end in kept], dim=-1)
