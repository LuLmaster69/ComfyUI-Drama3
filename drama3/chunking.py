"""Long text for Drama 3, as fish-speech's `fish_omni/chunking.py` (modeled on Fish Audio's hosted server).

Text is cut into chunks of about CHUNK_BYTES UTF-8 bytes with at most CHUNK_MAX_SPEAKERS speakers, preferring
sentence ends, then clause marks, then spaces; tags such as `<|speaker:1|>`, `[laughs]` or `<whisper>...</whisper>`
are never split. Each chunk is generated on its own; the first chunk and the latest one go back into the next
prompt as finished turns, which keeps the voice consistent across chunks.
"""
import re

from .prompt import LINE_SEPARATOR, SPEAKER_TAG, WHITE_SPACE, speaker_of, with_speaker_tag

CHUNK_BYTES = 300
CHUNK_MAX_SPEAKERS = 3
SENTENCE_ENDS = frozenset(".!?;。！？；…\n")
CLAUSE_MARKS = frozenset(",:，：、")
SPACES = frozenset(" ")
PROTECTED_SPAN = re.compile(r"<\|[^|>]*\|>|<([A-Za-z][\w-]*)\b[^>]*>.*?</\1>|\[[^\[\]]*\]", re.S)


def utf8_len(text):
    return len(text.encode("utf-8"))


def protected_pieces(text):
    pieces, last = [], 0
    for match in PROTECTED_SPAN.finditer(text):
        if match.start() > last:
            pieces.append((text[last:match.start()], False))
        pieces.append((match.group(0), True))
        last = match.end()
    if last < len(text):
        pieces.append((text[last:], False))
    return pieces


def breaks_at(text, index, marks):
    # keep decimals such as 3.14 in one piece
    return text[index] in marks and not (text[index] == "." and 0 < index < len(text) - 1
                                         and text[index - 1].isdigit() and text[index + 1].isdigit())


def break_at_marks(pieces, length, marks):
    """Split the free pieces longer than `length` bytes after each mark, keeping it."""
    out = []
    for text, protected in pieces:
        if protected or utf8_len(text) <= length:
            out.append((text, protected))
            continue
        start = 0
        for index in range(len(text)):
            if breaks_at(text, index, marks):
                out.append((text[start:index + 1], False))
                start = index + 1
        if start < len(text):
            out.append((text[start:], False))
    return out


def break_by_length(pieces, length):
    out = []
    for text, protected in pieces:
        if protected or utf8_len(text) <= length:
            out.append((text, protected))
            continue
        current = ""
        for char in text:
            if current and utf8_len(current + char) > length:
                out.append((current, False))
                current = ""
            current += char
        out.append((current, False))
    return out


def split_by_length(text, length=CHUNK_BYTES):
    if not text.strip(WHITE_SPACE):
        return []
    if utf8_len(text) <= length:
        return [text]
    pieces = break_at_marks(protected_pieces(text), 0, SENTENCE_ENDS)
    for marks in (CLAUSE_MARKS, SPACES):
        pieces = break_at_marks(pieces, length, marks)
    chunks, current = [], ""
    for piece, _ in break_by_length(pieces, length):
        if current and utf8_len(current + piece) > length:
            chunks.append(current)
            current = ""
        current += piece
    chunks.append(current)
    return [chunk.strip(WHITE_SPACE) for chunk in chunks if chunk.strip(WHITE_SPACE)]


def split_text(text, max_bytes=CHUNK_BYTES, max_speakers=CHUNK_MAX_SPEAKERS):
    """Generation chunks in order. Tagged text is split per speaker segment and regrouped: a chunk ends when the
    next segment would bring in one speaker too many or exceed `max_bytes`, so A, B, A can share a chunk. A chunk's
    segments are lines joined by <|audio_separator|>, the layout Drama 3 learned dialogue in (report, Figure 11)."""
    if SPEAKER_TAG.search(text) is None:
        return split_by_length(text, max_bytes)
    parts = SPEAKER_TAG.split(text)
    segments = [("0", parts[0])] if parts[0].strip(WHITE_SPACE) else []
    segments += [(speaker, body) for speaker, body in zip(parts[1::2], parts[2::2]) if body.strip(WHITE_SPACE)]
    tagged = [(speaker, f"<|speaker:{speaker}|> {piece.strip(WHITE_SPACE)}") for speaker, body in segments
              for piece in split_by_length(body, max_bytes)]
    chunks, batch, speakers, size = [], [], [], 0
    for speaker, segment in tagged:
        new_speaker = speaker not in speakers
        if (new_speaker and len(speakers) >= max_speakers) or (batch and size + utf8_len(segment) > max_bytes):
            chunks.append(LINE_SEPARATOR.join(batch))
            batch, speakers, size = [], [], 0
            new_speaker = True
        batch.append(segment)
        if new_speaker:
            speakers.append(speaker)
        size += utf8_len(segment)
    if batch:
        chunks.append(LINE_SEPARATOR.join(batch))
    return chunks


def by_speaker(chunks, max_bytes=CHUNK_BYTES):
    """One-speaker chunks (split_text with max_speakers 1) regrouped into chunks of one speaker's lines, speaker by speaker
    in the order they first speak: (chunks, the script position of each chunk's lines)."""
    lines = [with_speaker_tag(line) for chunk in chunks for line in chunk.split(LINE_SEPARATOR)]
    grouped, positions = [], []
    for speaker in dict.fromkeys(map(speaker_of, lines)):
        mine = [index for index, line in enumerate(lines) if speaker_of(line) == speaker]
        for chunk in split_text("\n".join(lines[index] for index in mine), max_bytes, 1):
            count = chunk.count(LINE_SEPARATOR) + 1
            grouped.append(chunk)
            positions.append(mine[:count])
            mine = mine[count:]
    return grouped, positions


def chunk_lines(chunk, index, frames, first_frame, config, separator_id):
    """The lines of one chunk's take, cut at the <|audio_separator|> rows it said: {speaker, text, chunk, start, end}
    with start and end in audio frames counted from `first_frame`. A take that said fewer separators than the chunk has
    lines keeps the rest in its last line; one that said more joins its extra pieces to its last line."""
    texts = chunk.split(LINE_SEPARATOR)
    bounds, start, count = [], first_frame, 0
    for token in frames[:, 0].tolist():
        if config.semantic_start_token_id <= token <= config.semantic_end_token_id:
            count += 1
        elif token == separator_id:
            bounds.append((start, first_frame + count))
            start = first_frame + count
    bounds.append((start, first_frame + count))
    if len(bounds) > len(texts):
        bounds = bounds[:len(texts) - 1] + [(bounds[len(texts) - 1][0], bounds[-1][1])]
    if len(texts) > len(bounds):
        texts = texts[:len(bounds) - 1] + ["\n".join(texts[len(bounds) - 1:])]
    lines = []
    for text, (start, end) in zip(texts, bounds):
        tag = SPEAKER_TAG.match(text.strip(WHITE_SPACE))
        lines.append({"speaker": int(tag.group(1)) if tag else 0, "text": text.strip(WHITE_SPACE)[tag.end() if tag else 0:].strip(WHITE_SPACE),
                      "chunk": index, "start": start, "end": end})
    return lines
