"""The Drama 3 Editor's model of a take: an analysis of the decoded audio (lines, words, loudness), a plan that turns the
editor's settings and manual edits into an edit list, and the rendering of that list.

Times are source seconds unless a name says otherwise. The plan works like the silence tools of audio editors
(Audacity's Truncate Silence, Descript's Shorten word gaps): a pause longer than a limit is shortened to a target by
removing its middle, so the decay of the last word and the breath before the next stay. Gaps between lines have their
own range, so lines that run into each other get room and long gaps between speakers are tightened.
"""
import bisect
import json
import re

import torch

from ..qwen3_asr.align import align, words

ENVELOPE_STEP = 0.01
MIN_PAUSE = 0.1
FADE = 0.004
# non-spoken spans of a line: [cues], <|tokens|>, and the markers (not the words) of <scoped>...</scoped> tags
NOT_SPOKEN = re.compile(r"\[[^\[\]]*\]|<\|[^|>]*\|>|</?[A-Za-z][\w-]*\b[^>]*>")
DEFAULTS = {
    "threshold_db": -50.0,
    "trim_start": True, "lead": 0.05,
    "trim_end": True, "tail": 0.15,
    "pause_longer": 0.75, "pause_to": 0.75,
    "gap_min": 0.3, "gap_max": 1.0,
    "keep": [], "cuts": [], "inserts": [],
    "speakers": {},
    "normalize": False,
}


def parse_edits(text):
    """The editor's settings and edits (its hidden widget's JSON) over the defaults."""
    edits = dict(DEFAULTS)
    if text:
        edits.update(json.loads(text))
    return edits


def tokens(text):
    """Display tokens of a line: {"text", "spoken"}; a director's note ("Direction: ...\\nLine: ...") is not spoken."""
    out = []
    note = re.match(r"\s*Direction:(.*?)\n\s*Line:", text, re.S)
    if note:
        out.append({"text": note.group(0).strip(), "spoken": False})
        text = text[note.end():]
    position = 0
    for match in NOT_SPOKEN.finditer(text):
        out += [{"text": word, "spoken": True} for word in text[position:match.start()].split()]
        out.append({"text": match.group(0), "spoken": False})
        position = match.end()
    out += [{"text": word, "spoken": True} for word in text[position:].split()]
    return out


def analyze(waveform, sample_rate, lines, aligner=None):
    """The take's analysis for the editor: duration, a loudness envelope (dB per ENVELOPE_STEP), and its lines in
    seconds with display tokens, each spoken one timed when an aligner is given. `lines` are Generate's, in frames."""
    frame = 2048 / sample_rate
    step = int(ENVELOPE_STEP * sample_rate)
    power = waveform[:waveform.shape[-1] // step * step].reshape(-1, step).pow(2).mean(-1)
    envelope = [round(value, 1) for value in (10 * torch.log10(power + 1e-12)).tolist()]
    out = []
    for line in lines:
        start, end = line["start"] * frame, line["end"] * frame
        line_tokens = tokens(line["text"])
        if aligner is not None and end > start:
            spoken = [token for token in line_tokens if token["spoken"]]
            timed = iter(align(aligner, waveform[None, int(start * sample_rate):int(end * sample_rate)], sample_rate,
                               " ".join(token["text"] for token in spoken)))
            for token in spoken:
                units = [next(timed) for _ in words(token["text"])]
                if units:
                    token["start"], token["end"] = round(start + units[0][1], 3), round(start + units[-1][2], 3)
        out.append({"speaker": line["speaker"], "chunk": line["chunk"], "start": round(start, 3), "end": round(end, 3),
                    "text": line["text"], "tokens": line_tokens})
    return {"sample_rate": sample_rate, "duration": round(waveform.shape[-1] / sample_rate, 3), "envelope_step": ENVELOPE_STEP,
            "envelope": envelope, "lines": out, "speakers": sorted({line["speaker"] for line in out}),
            "aligned": aligner is not None}


def pauses(analysis, threshold_db):
    """Quiet runs of at least MIN_PAUSE: [start, end] seconds."""
    step, runs, start = analysis["envelope_step"], [], None
    for index, value in enumerate(analysis["envelope"] + [0.0]):
        quiet = value < threshold_db
        if quiet and start is None:
            start = index
        elif not quiet and start is not None:
            if (index - start) * step >= MIN_PAUSE:
                runs.append([round(start * step, 3), round(min(index * step, analysis["duration"]), 3)])
            start = None
    return runs


def plan(analysis, edits):
    """The edit list for `edits`: {"pieces": [{"src": [a, b], "line": i} or {"gap": seconds, "line": i}],
    "pauses": [{"id", "start", "end", "kind", "action", "to"}], "duration": seconds after editing}."""
    duration, lines = analysis["duration"], analysis["lines"]
    boundaries = [line["end"] for line in lines[:-1]]
    removed, inserted, found = [], [], []
    keep = set(edits["keep"])
    covered = set()
    sound = [run for run in pauses(analysis, edits["threshold_db"])]
    for start, end in sound:
        length, pause_id = end - start, f"{start:.2f}"
        boundary = next((b for b in boundaries if start - 0.05 <= b <= end + 0.05), None)
        kind = "lead" if start <= 0 else "tail" if end >= duration else "between" if boundary is not None else "inside"
        action, to = "kept", length
        if pause_id in keep:
            action = "kept by you"
        elif kind == "lead" and edits["trim_start"]:
            to = min(length, edits["lead"])
        elif kind == "tail" and edits["trim_end"]:
            to = min(length, edits["tail"])
        elif kind == "between":
            covered.add(boundary)
            to = min(max(length, edits["gap_min"]), edits["gap_max"])
        elif kind == "inside" and length > edits["pause_longer"]:
            to = edits["pause_to"]
        if to < length - 1e-3:
            action = "shortened"
            if kind == "lead":
                removed.append([start, end - to])
            elif kind == "tail":
                removed.append([start + to, end])
            else:
                removed.append([start + to / 2, end - to / 2])
        elif to > length + 1e-3:
            action = "lengthened"
            inserted.append([(start + end) / 2, to - length])
        found.append({"id": pause_id, "start": start, "end": end, "kind": kind, "action": action, "to": round(to, 3)})
    # lines that run straight into each other get the minimum gap
    for boundary in boundaries:
        if boundary not in covered and edits["gap_min"] > 0:
            inserted.append([boundary, edits["gap_min"]])
    removed += [list(cut) for cut in edits["cuts"]]
    inserted += [list(insert) for insert in edits["inserts"]]
    pieces = cut_pieces(duration, removed, inserted, [line["start"] for line in lines])
    total = sum(piece["src"][1] - piece["src"][0] if "src" in piece else piece["gap"] for piece in pieces)
    return {"pieces": pieces, "pauses": found, "duration": round(total, 3)}


def cut_pieces(duration, removed, inserted, line_starts):
    """Source pieces between the removed ranges, split at line starts, with each inserted gap at its time (or at the next
    kept audio, when its time was removed)."""
    kept, position = [], 0.0
    for a, b in sorted([max(0.0, a), min(duration, b)] for a, b in removed if b > a):
        if a > position:
            kept.append([position, a])
        position = max(position, b)
    if position < duration:
        kept.append([position, duration])

    def line_at(t):
        return max(0, bisect.bisect_right(line_starts, t + 1e-6) - 1)

    pending, pieces = sorted(inserted), []
    for a, b in kept:
        points = sorted({t for t in line_starts if a < t < b} | {t for t, _ in pending if a < t < b})
        for x, y in zip([a, *points], [*points, b]):
            while pending and pending[0][0] <= x + 1e-6:
                pieces.append({"gap": round(pending.pop(0)[1], 3), "line": line_at(x)})
            pieces.append({"src": [round(x, 4), round(y, 4)], "line": line_at(x)})
    pieces += [{"gap": round(seconds, 3), "line": line_at(t)} for t, seconds in pending]
    return pieces


def render(waveform, analysis, pieces, edits):
    """The edited take [samples]: source pieces with FADE ramps where they do not continue the previous one, silence for
    gaps, each speaker's gain and mute, then an optional peak normalization to -1 dBFS."""
    rate = analysis["sample_rate"]
    lines = analysis["lines"]
    fade = max(1, int(FADE * rate))

    def joins(first, second):
        return first is not None and second is not None and "src" in first and "src" in second and first["src"][1] == second["src"][0]

    out = []
    for index, piece in enumerate(pieces):
        if "gap" in piece:
            out.append(torch.zeros(int(round(piece["gap"] * rate))))
            continue
        a, b = int(round(piece["src"][0] * rate)), int(round(piece["src"][1] * rate))
        audio = waveform[a:b].clone()
        speaker = lines[piece["line"]]["speaker"] if piece["line"] is not None and lines else 0
        setting = edits["speakers"].get(str(speaker), {})
        if setting.get("mute"):
            audio.zero_()
        elif setting.get("gain_db"):
            audio *= 10 ** (setting["gain_db"] / 20)
        if len(audio) > 2 * fade:
            if not joins(pieces[index - 1] if index else None, piece):
                audio[:fade] *= torch.linspace(0, 1, fade)
            if not joins(piece, pieces[index + 1] if index + 1 < len(pieces) else None):
                audio[-fade:] *= torch.linspace(1, 0, fade)
        out.append(audio)
    result = torch.cat(out) if out else torch.zeros(1)
    if edits["normalize"]:
        peak = result.abs().max().item()
        if peak > 0:
            result = result * (10 ** (-1 / 20) / peak)
    return result.clamp(-1, 1)

