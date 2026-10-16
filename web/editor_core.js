// Drama 3 Editor core: a take on a timeline, the compact preview and the large view. No dependencies; the ComfyUI
// node (drama3_editor.js) and the standalone app hand it their storage and server:
//
//   const view = new EditorView({ readEdits, writeEdits, apply, plan: (analysis, edits) => Promise<plan>, applyText });
//   const preview = new TakePreview(view, { onOpen, emptyText });   // preview.root goes in the page
//   await loadTake(view, analysisUrl, audioUrl);                    // the take as the Python side analyzed it
//   new EditorWindow(view, onClose);                                // the large view, inset over the page
//
// Python owns the edit logic (drama3/editor.py): the view asks `plan` for the edit list whenever a setting changes,
// plays it with Web Audio from the unedited take, and draws it. `apply` renders the same list on the Python side.
//
// Layout: captions that follow playback (the line before, the line being spoken word by word, the line after), one
// track per voice as in ElevenLabs Studio and multitrack editors, the play controls under the tracks, silence presets
// in the spirit of Descript's "shorten word gaps" and Audacity's Truncate Silence, and a list of things to check.

const STYLE_ID = "drama3-editor-style";
const PALETTE = ["#4fc3f7", "#ffb74d", "#81c784", "#f06292", "#ba68c8", "#4db6ac", "#e57373", "#aed581",
  "#7986cb", "#ffd54f", "#a1887f", "#90a4ae", "#ff8a65", "#9575cd", "#4dd0e1", "#dce775"];
const DEFAULTS = {
  threshold_db: -50, trim_start: true, lead: 0.05, trim_end: true, tail: 0.15,
  pause_longer: 0.75, pause_to: 0.75, gap_min: 0.3, gap_max: 1.0,
  keep: [], cuts: [], inserts: [], speakers: {}, normalize: false,
};
// "As generated" turns every automatic change off; "Natural" is DEFAULTS
const PRESETS = [
  { id: "generated", name: "As generated", hint: "Change nothing",
    values: { trim_start: false, trim_end: false, pause_longer: 60, pause_to: 60, gap_min: 0, gap_max: 60 } },
  { id: "natural", name: "Natural", hint: "Trim the ends, pauses at most 0.75 s, lines 0.3–1 s apart",
    values: { trim_start: true, lead: 0.05, trim_end: true, tail: 0.15, pause_longer: 0.75, pause_to: 0.75, gap_min: 0.3, gap_max: 1.0 } },
  { id: "tight", name: "Tight", hint: "Pauses at most 0.4 s, lines 0.2–0.6 s apart",
    values: { trim_start: true, lead: 0.05, trim_end: true, tail: 0.1, pause_longer: 0.4, pause_to: 0.4, gap_min: 0.2, gap_max: 0.6 } },
  { id: "custom", name: "Custom", hint: "Set every value yourself" },
];
const LONG_SILENCE = 1.5;  // seconds; longer silences are listed under Things to check
const PEAK_STEP = 0.005;   // seconds per waveform bucket
const TRACK_KEY = "drama3.editor.oneTrack";
const FADE = 0.004;
const HISTORY = 200;
// what "apply" means to the host: the ComfyUI node runs the workflow again
const APPLY_IN_COMFYUI = {
  label: "✓ Use these edits",
  title: "Run the workflow with these edits: only this node runs again, the take itself is reused (Ctrl+Enter)",
  done: "Running the workflow with your edits",
};
const EXPAND_ICON = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.4" stroke-linecap="round"><path d="M14 4h6v6M10 20H4v-6M20 4l-7 7M4 20l7-7"/></svg>';

const CSS = `
.d3e-node { position: relative; width: 100%; height: 100%; min-height: 180px; display: flex; flex-direction: column;
  gap: 5px; padding: 6px; box-sizing: border-box; background: #16161a; border-radius: 6px; color: #ddd;
  font: 12px/1.35 system-ui, sans-serif; }
.d3e-node canvas { width: 100%; flex: 1 1 0; min-height: 40px; border-radius: 4px; background: #0f0f12; cursor: pointer; }
.d3e-row { display: flex; align-items: center; gap: 6px; min-width: 0; }
.d3e-grow { flex: 1; min-width: 0; }
.d3e-time { font-variant-numeric: tabular-nums; color: #cfd8dc; white-space: nowrap; padding: 0 4px; }
.d3e-dim { color: #8d95a6; }
.d3e-btn { display: inline-flex; align-items: center; gap: 5px; background: #2a2a31; color: #e6e6e6; border: 1px solid #3a3a44;
  border-radius: 5px; padding: 3px 9px; font: inherit; cursor: pointer; white-space: nowrap; }
.d3e-btn svg { width: 12px; height: 12px; }
.d3e-btn:hover { background: #34343d; }
.d3e-btn.d3e-on { background: #3b4a6b; border-color: #5a77b3; }
.d3e-btn.d3e-primary { background: #2f6fd6; border-color: #2f6fd6; color: #fff; }
.d3e-btn.d3e-primary:hover { background: #3d7ff0; }
.d3e-btn:disabled { opacity: .45; cursor: default; }
.d3e-status { position: absolute; inset: 0; display: flex; align-items: center; justify-content: center; text-align: center;
  padding: 12px; color: #888; background: #16161a; border-radius: 6px; }
.d3e-cap { position: relative; overflow: hidden; min-width: 0; }
.d3e-node .d3e-cap { flex: 1 1 0; min-height: 58px; }
.d3e-reel { position: absolute; left: 0; right: 0; top: 0; display: flex; flex-direction: column; align-items: center; gap: 3px;
  transition: transform .4s ease; }
.d3e-cap-row { max-width: 86%; text-align: center; font-size: 11px; line-height: 1.4; cursor: pointer; opacity: 0;
  transition: opacity .4s ease, transform .4s ease; transform-origin: center; }
.d3e-cap-row.d3e-d0 { opacity: 1; transform: scale(1.14); cursor: default; }
.d3e-cap-row.d3e-d1 { opacity: .5; }
.d3e-cap-row.d3e-d2 { opacity: .3; }
.d3e-cap-row.d3e-d3 { opacity: .16; }
.d3e-cap-big { width: 100%; height: 100%; }
.d3e-cap-big .d3e-reel { gap: 10px; }
.d3e-cap-big .d3e-cap-row { font-size: 17px; max-width: 78%; }
.d3e-voice { font-weight: 600; margin-right: 6px; }
.d3e-word { border-radius: 3px; padding: 0 1px; cursor: pointer; color: #9aa0ad; }
.d3e-cap-row:not(.d3e-d0) .d3e-word, .d3e-untimed .d3e-word, .d3e-word.d3e-said { color: #f0f0f0; }
.d3e-cap-row:not(.d3e-d0) .d3e-word { background: none; cursor: inherit; }
.d3e-word.d3e-now { background: #f5c542; color: #111; }
.d3e-word.d3e-cut { text-decoration: line-through; color: #666; }
.d3e-cue { color: #8a8fa0; font-style: italic; }
.d3e-overlay { position: fixed; inset: 0; z-index: 10000; background: rgba(8, 8, 12, .55); padding: 3.5vh 3vw;
  box-sizing: border-box; display: flex; }
.d3e-window { position: relative; flex: 1; min-width: 0; min-height: 0; display: grid; outline: none; overflow: hidden;
  grid-template-columns: minmax(0, 1fr) 340px; grid-template-rows: auto auto minmax(120px, 1fr) auto auto auto;
  grid-template-areas: "bar bar" "presets presets" "captions checks" "tracks checks" "transport checks" "foot foot";
  background: #16161b; color: #ddd; border: 1px solid #34343e; border-radius: 10px; box-shadow: 0 12px 48px rgba(0, 0, 0, .6);
  font: 13px/1.4 system-ui, sans-serif; }
.d3e-bar { grid-area: bar; display: flex; align-items: center; gap: 8px; padding: 8px 12px; border-bottom: 1px solid #2c2c34;
  background: #1b1b21; flex-wrap: wrap; }
.d3e-title { font-weight: 600; }
.d3e-spacer { flex: 1; }
.d3e-sep { width: 1px; height: 20px; background: #33333b; margin: 0 2px; }
.d3e-presets { grid-area: presets; padding: 10px 12px 12px; border-bottom: 1px solid #2c2c34; }
.d3e-step { font-weight: 600; margin-bottom: 8px; color: #e6e6e6; }
.d3e-step small { font-weight: 400; color: #8d95a6; margin-left: 8px; }
.d3e-cards { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 8px; }
.d3e-card { border: 1px solid #34343e; border-radius: 8px; padding: 7px 10px; cursor: pointer; background: #1b1b21; }
.d3e-card:hover { border-color: #4a4a56; }
.d3e-card.d3e-on { border: 2px solid #2f6fd6; padding: 6px 9px; background: #1d2333; }
.d3e-card b { display: block; font-weight: 600; }
.d3e-card span { color: #8d95a6; font-size: 12px; }
.d3e-custom { display: grid; grid-template-columns: repeat(auto-fit, minmax(230px, 1fr)); gap: 10px; margin-top: 10px; }
.d3e-group { background: #1b1b21; border: 1px solid #2c2c34; border-radius: 8px; padding: 9px 11px; display: flex;
  flex-direction: column; gap: 7px; }
.d3e-group h5 { margin: 0; font-size: 12px; font-weight: 600; color: #e6e6e6; }
.d3e-group small { color: #8d95a6; font-size: 11px; line-height: 1.35; }
.d3e-field { display: flex; align-items: center; justify-content: space-between; gap: 10px; min-height: 24px; }
.d3e-unit { display: inline-flex; align-items: center; gap: 5px; color: #8d95a6; white-space: nowrap; }
.d3e-unit input[type=range] { width: 96px; }
.d3e-toggle { display: flex; align-items: center; gap: 7px; cursor: pointer; }
.d3e-window input[type=number]:disabled { opacity: .4; }
.d3e-window input[type=number] { width: 58px; background: #111; color: #ddd; border: 1px solid #333; border-radius: 4px; padding: 2px 4px; }
.d3e-window input[type=text] { width: 100%; box-sizing: border-box; background: #111; color: #ddd; border: 1px solid #5a77b3;
  border-radius: 4px; padding: 1px 4px; font: inherit; }
.d3e-captions { grid-area: captions; position: relative; min-height: 0; overflow: hidden; }
.d3e-captions > .d3e-cap { position: absolute; inset: 0; }
.d3e-tracks { grid-area: tracks; display: grid; grid-template-columns: 190px minmax(0, 1fr); border-top: 1px solid #2c2c34;
  background: #0f0f12; }
.d3e-heads { background: #18181d; border-right: 1px solid #2c2c34; }
.d3e-ruler-gap { border-bottom: 1px solid #26262d; box-sizing: border-box; display: flex; align-items: center; padding: 0 6px; }
.d3e-chips { display: flex; flex-wrap: wrap; gap: 4px 10px; }
.d3e-chip { display: inline-flex; align-items: center; gap: 5px; }
.d3e-head { display: flex; flex-direction: column; justify-content: center; gap: 4px; padding: 0 8px; box-sizing: border-box;
  border-bottom: 1px solid #202027; }
.d3e-swatch { flex: none; display: inline-block; width: 10px; height: 10px; border-radius: 3px; }
.d3e-name { flex: 1; min-width: 0; cursor: text; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.d3e-name small { color: #8d95a6; margin-left: 4px; }
.d3e-head input[type=range] { flex: 1; min-width: 0; }
.d3e-mini { padding: 1px 7px; font-size: 11px; }
.d3e-scroll { overflow-x: auto; overflow-y: hidden; position: relative; }
.d3e-scroll canvas { display: block; }
.d3e-transport { grid-area: transport; display: flex; align-items: center; gap: 6px; padding: 7px 12px; border-top: 1px solid #2c2c34;
  background: #1b1b21; flex-wrap: wrap; }
.d3e-checks { grid-area: checks; border-left: 1px solid #2c2c34; background: #141418; overflow: auto; padding: 10px 12px; }
.d3e-checks .d3e-step { margin-top: 4px; }
.d3e-item { display: grid; grid-template-columns: 18px minmax(0, 1fr) auto; gap: 6px; align-items: start; padding: 7px 4px;
  border-bottom: 1px solid #23232a; cursor: pointer; border-radius: 4px; }
.d3e-item:hover { background: #1d1d24; }
.d3e-item small { display: block; color: #8d95a6; }
.d3e-acts { display: flex; flex-direction: column; gap: 4px; align-items: flex-end; }
.d3e-warn { color: #ffb74d; }
.d3e-empty { color: #8d95a6; padding: 6px 4px 12px; }
.d3e-window.d3e-narrow { grid-template-columns: minmax(0, 1fr); grid-template-rows: auto auto minmax(120px, 1fr) auto auto auto auto;
  grid-template-areas: "bar" "presets" "captions" "tracks" "transport" "checks" "foot"; }
.d3e-window.d3e-narrow .d3e-checks { border-left: none; border-top: 1px solid #2c2c34; max-height: 30vh; }
.d3e-window.d3e-narrow .d3e-cards { grid-template-columns: repeat(2, minmax(0, 1fr)); }
.d3e-window.d3e-narrow .d3e-cap-big .d3e-cap-row { font-size: 15px; }
.d3e-foot { grid-area: foot; display: flex; gap: 16px; align-items: center; padding: 5px 12px; border-top: 1px solid #2c2c34;
  background: #18181d; color: #8d95a6; font-size: 12px; flex-wrap: wrap; }
.d3e-foot b { color: #cfd8dc; font-weight: 500; }
.d3e-key { display: inline-flex; align-items: center; gap: 5px; }
.d3e-key i { width: 10px; height: 10px; border-radius: 2px; display: inline-block; }
.d3e-toast { position: absolute; left: 50%; bottom: 46px; transform: translateX(-50%); z-index: 3; padding: 6px 12px; border-radius: 8px;
  background: rgba(20, 20, 24, .94); border: 1px solid #3f9a6c; pointer-events: none; opacity: 0; transition: opacity .2s; }
.d3e-toast.d3e-show { opacity: 1; }
.d3e-help { position: absolute; right: 16px; top: 46px; z-index: 4; width: 380px; background: #1c1c22; border: 1px solid #3a3a44;
  border-radius: 8px; padding: 12px 14px; box-shadow: 0 10px 30px rgba(0, 0, 0, .5); font-size: 12px; }
.d3e-help h4 { margin: 0 0 6px; font-size: 13px; }
.d3e-help table { width: 100%; border-collapse: collapse; margin-bottom: 10px; }
.d3e-help td { padding: 2px 4px; vertical-align: top; }
.d3e-help td:first-child { color: #f5c542; white-space: nowrap; font-family: ui-monospace, monospace; }
`;

function injectStyle() {
  if (document.getElementById(STYLE_ID)) return;
  const style = document.createElement("style");
  style.id = STYLE_ID;
  style.textContent = CSS;
  document.head.append(style);
}

function el(tag, attrs = {}, ...children) {
  const node = document.createElement(tag);
  for (const [key, value] of Object.entries(attrs)) {
    if (key === "class") node.className = value;
    else if (key === "style") node.style.cssText = value;
    else if (key.startsWith("on")) node.addEventListener(key.slice(2), value);
    else if (value !== undefined && value !== null && value !== false) node.setAttribute(key, value === true ? "" : value);
  }
  node.append(...children.flat().filter((c) => c !== null && c !== undefined && c !== false));
  return node;
}

function icon(svg) {
  const holder = document.createElement("span");
  holder.innerHTML = svg;
  return holder.firstChild;
}

function fmt(seconds, precise = true) {
  const s = Math.max(0, seconds);
  const m = Math.floor(s / 60);
  const rest = s - m * 60;
  return `${m}:${rest.toFixed(precise ? 2 : 0).padStart(precise ? 5 : 2, "0")}`;
}

function speakerColor(speaker) {
  return PALETTE[speaker % PALETTE.length];
}

// view preferences live in the browser; storage can be missing (private windows, blocked site data)
function readSetting(key) {
  try { return localStorage.getItem(key); } catch { return null; }
}

function writeSetting(key, value) {
  try { localStorage.setItem(key, value); } catch { /* the preference lasts for this page only */ }
}

// --------------------------------------------------------------------------------------------------------------
// Model: analysis + edits -> plan (from Python) -> time maps for playback and drawing

class Take {
  constructor(analysis, buffer) {
    this.analysis = analysis;
    this.buffer = buffer;
    this.lines = analysis.lines;
    this.speakers = analysis.speakers.length ? analysis.speakers : [0];
    this.texts = this.lines.map((line) => line.tokens.filter((t) => t.spoken).map((t) => t.text).join(" "));
    this.peaks = Take.peaks(buffer);
  }

  static peaks(buffer) {
    const data = buffer.getChannelData(0);
    const step = Math.max(1, Math.round(PEAK_STEP * buffer.sampleRate));
    const count = Math.ceil(data.length / step);
    const peaks = new Float32Array(count * 2);
    for (let i = 0; i < count; i++) {
      let lo = 0, hi = 0;
      const end = Math.min(data.length, (i + 1) * step);
      for (let j = i * step; j < end; j++) {
        const v = data[j];
        if (v < lo) lo = v;
        if (v > hi) hi = v;
      }
      peaks[2 * i] = lo;
      peaks[2 * i + 1] = hi;
    }
    return peaks;
  }

  // the unedited take as a plan: every line once, nothing removed
  originalPlan() {
    const pieces = this.lines.length
      ? this.lines.map((line, i) => ({ src: [line.start, i + 1 < this.lines.length ? this.lines[i + 1].start : this.analysis.duration], line: i }))
      : [{ src: [0, this.analysis.duration], line: 0 }];
    return { pieces, pauses: [], duration: this.analysis.duration };
  }
}

// Edited-time layout of a plan: each piece's start in edited seconds, and maps between source and edited time.
class Layout {
  constructor(plan, take) {
    this.plan = plan;
    this.take = take;
    this.items = [];
    let t = 0;
    for (const piece of plan.pieces) {
      const length = piece.src ? piece.src[1] - piece.src[0] : piece.gap;
      this.items.push({ ...piece, start: t, end: t + length, speaker: this.speakerOf(piece) });
      t += length;
    }
    this.duration = t;
  }

  speakerOf(piece) {
    const line = this.take.lines[piece.line];
    return line ? line.speaker : 0;
  }

  // edited seconds -> source seconds (a gap maps to where its next audio starts)
  toSource(t) {
    for (const item of this.items) {
      if (t < item.end || item === this.items[this.items.length - 1]) {
        if (item.src) return item.src[0] + Math.min(Math.max(0, t - item.start), item.src[1] - item.src[0]);
        const next = this.items.find((i) => i.src && i.start >= item.end);
        return next ? next.src[0] : this.take.analysis.duration;
      }
    }
    return 0;
  }

  // source seconds -> edited seconds, or null when that moment was cut
  toEdited(s) {
    for (const item of this.items) {
      if (item.src && s >= item.src[0] && s < item.src[1]) return item.start + (s - item.src[0]);
    }
    return null;
  }

  // the cuts between pieces, at their edited positions, with the source length they removed
  cuts() {
    const out = [];
    let previous = null;
    for (const item of this.items) {
      if (!item.src) continue;
      if (previous && Math.abs(previous.src[1] - item.src[0]) > 1e-3 && item.src[0] > previous.src[1]) {
        out.push({ at: item.start, removed: item.src[0] - previous.src[1], from: previous.src[1], to: item.src[0] });
      }
      previous = item;
    }
    const first = this.items.find((i) => i.src);
    if (first && first.src[0] > 1e-3) out.unshift({ at: first.start, removed: first.src[0], from: 0, to: first.src[0] });
    return out;
  }
}

// --------------------------------------------------------------------------------------------------------------
// Playback of a layout with Web Audio: one buffer source per piece, short ramps at every join

class Player {
  constructor(onEnd) {
    this.ctx = null;
    this.nodes = [];
    this.playing = false;
    this.startedAt = 0;
    this.from = 0;
    this.until = null;
    this.onEnd = onEnd;
  }

  context() {
    if (!this.ctx) this.ctx = new (window.AudioContext || window.webkitAudioContext)();
    return this.ctx;
  }

  position() {
    if (!this.playing) return this.from;
    return this.from + (this.ctx.currentTime - this.startedAt);
  }

  stop(keepPosition = true) {
    if (keepPosition) this.from = this.position();
    for (const n of this.nodes) { try { n.stop(); } catch { /* already stopped */ } }
    this.nodes = [];
    this.playing = false;
    clearTimeout(this.endTimer);
  }

  // play `layout` from edited time `from` to `until` (or its end); gains maps speaker -> linear gain (0 mutes)
  play(layout, buffer, from, until, gains) {
    this.stop(false);
    const ctx = this.context();
    ctx.resume();
    const end = Math.min(until ?? layout.duration, layout.duration);
    if (from >= end - 1e-3) from = 0;
    const start = ctx.currentTime + 0.04;
    for (const item of layout.items) {
      if (!item.src || item.end <= from || item.start >= end) continue;
      const offset = Math.max(0, from - item.start);
      const length = Math.min(item.end, end) - item.start - offset;
      if (length <= 0) continue;
      const source = ctx.createBufferSource();
      source.buffer = buffer;
      const gain = ctx.createGain();
      const level = gains(item.speaker);
      const when = start + Math.max(0, item.start - from);
      gain.gain.setValueAtTime(0, when);
      gain.gain.linearRampToValueAtTime(level, when + FADE);
      gain.gain.setValueAtTime(level, Math.max(when + FADE, when + length - FADE));
      gain.gain.linearRampToValueAtTime(0, when + length);
      source.connect(gain).connect(ctx.destination);
      source.start(when, item.src[0] + offset, length);
      this.nodes.push(source);
    }
    this.from = from;
    this.startedAt = start;
    this.until = end;
    this.playing = true;
    this.endTimer = setTimeout(() => { this.stop(false); this.from = end; this.onEnd?.(); }, (end - from) * 1000 + 80);
  }
}

// --------------------------------------------------------------------------------------------------------------
// The editor

class EditorView {
  constructor({ readEdits, writeEdits, apply, plan, applyText = APPLY_IN_COMFYUI }) {
    this.readEdits = readEdits;
    this.writeEdits = writeEdits;
    this.applyWorkflow = apply;
    this.requestPlan = plan;
    this.applyText = applyText;
    this.oneTrack = readSetting(TRACK_KEY) === "1";
    this.take = null;
    this.edits = { ...DEFAULTS };
    this.plan = null;
    this.layout = null;
    this.original = false;
    this.selection = null;      // [a, b] edited seconds
    this.loop = false;
    this.zoom = null;           // px per second; null = fit
    this.undoStack = [];
    this.redoStack = [];
    this.player = new Player(() => this.onPlaybackEnd());
    this.listeners = new Set();
    this.raf = null;
  }

  subscribe(fn) { this.listeners.add(fn); return () => this.listeners.delete(fn); }
  emit() { for (const fn of this.listeners) fn(); }

  async load(analysis, buffer) {
    this.player.stop(false);
    this.player.from = 0;
    this.take = new Take(analysis, buffer);
    this.edits = { ...DEFAULTS, ...this.readEdits() };
    this.undoStack = [];
    this.redoStack = [];
    this.selection = null;
    await this.replan();
  }

  async replan() {
    if (!this.take) return;
    const token = (this.planToken = (this.planToken || 0) + 1);
    const plan = await this.requestPlan(this.take.analysis, this.edits);
    if (token !== this.planToken) return;
    this.plan = plan;
    this.relayout();
  }

  relayout() {
    const wasPlaying = this.player.playing;
    const source = this.layout ? this.layout.toSource(this.player.position()) : 0;
    this.layout = new Layout(this.original ? this.take.originalPlan() : this.plan, this.take);
    const edited = this.layout.toEdited(source);
    this.player.stop(false);
    this.player.from = edited ?? Math.min(this.player.from, this.layout.duration);
    if (wasPlaying) this.play();
    this.emit();
  }

  // settings and edits: every change goes through here, with undo
  change(patch, { history = true } = {}) {
    if (history) {
      this.undoStack.push(JSON.stringify(this.edits));
      if (this.undoStack.length > HISTORY) this.undoStack.shift();
      this.redoStack = [];
    }
    this.edits = { ...this.edits, ...patch };
    this.writeEdits(this.edits);
    this.replan();
  }

  undo() {
    if (!this.undoStack.length) return;
    this.redoStack.push(JSON.stringify(this.edits));
    this.edits = JSON.parse(this.undoStack.pop());
    this.writeEdits(this.edits);
    this.replan();
  }

  redo() {
    if (!this.redoStack.length) return;
    this.undoStack.push(JSON.stringify(this.edits));
    this.edits = JSON.parse(this.redoStack.pop());
    this.writeEdits(this.edits);
    this.replan();
  }

  reset() {
    this.change({ ...DEFAULTS, cuts: this.edits.cuts, inserts: this.edits.inserts, keep: [] });
  }

  setOneTrack(on) {
    this.oneTrack = on;
    writeSetting(TRACK_KEY, on ? "1" : "0");
    this.emit();
  }

  setOriginal(on) {
    if (this.original === on) return;
    this.original = on;
    this.selection = null;
    this.relayout();
  }

  gain(speaker) {
    const s = this.edits.speakers[String(speaker)] || {};
    if (s.mute) return 0;
    return Math.pow(10, (s.gain_db || 0) / 20);
  }

  // "Joe (Voice 0)" once the voice has a name: voice N is Generate's slot voiceN and <|speaker:N|>
  voiceLabel(speaker) {
    const name = (this.edits.speakers[String(speaker)] || {}).name;
    return name ? `${name} (Voice ${speaker})` : `Voice ${speaker}`;
  }

  jumpLine(direction) {
    if (!this.layout) return;
    const position = this.player.position();
    const starts = this.take.lines.map((line) => this.layout.toEdited(line.start)).filter((t) => t !== null);
    const target = direction < 0 ? starts.filter((t) => t < position - 0.3).pop() ?? 0 : starts.find((t) => t > position + 0.05);
    if (target !== undefined) this.seek(target);
  }

  play(from = null, until = null) {
    if (!this.layout) return;
    const start = from ?? this.player.position();
    const stop = until ?? (this.loop && this.selection ? this.selection[1] : null);
    this.player.play(this.layout, this.take.buffer, start, stop, (s) => this.gain(s));
    this.tick();
    this.emit();
  }

  pause() {
    this.player.stop(true);
    this.emit();
  }

  toggle() {
    if (this.player.playing) this.pause();
    else if (this.selection && (this.player.from < this.selection[0] || this.player.from >= this.selection[1])) this.play(this.selection[0]);
    else this.play();
  }

  seek(t, { keepPlaying = true } = {}) {
    if (!this.layout) return;
    const target = Math.max(0, Math.min(t, this.layout.duration));
    if (this.player.playing && keepPlaying) this.play(target);
    else { this.player.stop(false); this.player.from = target; this.emit(); }
  }

  onPlaybackEnd() {
    if (this.loop && this.selection) { this.play(this.selection[0]); return; }
    this.emit();
  }

  tick() {
    cancelAnimationFrame(this.raf);
    const step = () => {
      this.emit();
      if (this.player.playing) this.raf = requestAnimationFrame(step);
    };
    this.raf = requestAnimationFrame(step);
  }

  // edits on a selection of the edited timeline, stored in source time
  cutSelection() {
    if (!this.selection || this.original) return;
    const [a, b] = this.selection;
    const ranges = [];
    for (const item of this.layout.items) {
      if (!item.src || item.end <= a || item.start >= b) continue;
      const s0 = item.src[0] + Math.max(0, a - item.start);
      const s1 = item.src[0] + Math.min(item.end, b) - item.start;
      if (s1 > s0) ranges.push([+s0.toFixed(4), +s1.toFixed(4)]);
    }
    if (!ranges.length) return;
    this.change({ cuts: [...this.edits.cuts, ...ranges] });
    this.selection = null;
    this.seek(a, { keepPlaying: false });
  }

  insertSilence(seconds) {
    if (this.original || !this.layout) return;
    const at = this.layout.toSource(this.selection ? this.selection[0] : this.player.position());
    this.change({ inserts: [...this.edits.inserts, [+at.toFixed(4), seconds]] });
  }

  removeEdit(kind, index) {
    const list = [...this.edits[kind]];
    list.splice(index, 1);
    this.change({ [kind]: list });
  }

  toggleKeep(pauseId) {
    const keep = new Set(this.edits.keep);
    if (keep.has(pauseId)) keep.delete(pauseId); else keep.add(pauseId);
    this.change({ keep: [...keep] });
  }

  setSpeaker(speaker, patch) {
    const speakers = { ...this.edits.speakers, [String(speaker)]: { ...(this.edits.speakers[String(speaker)] || {}), ...patch } };
    this.change({ speakers });
  }

  apply() {
    this.writeEdits(this.edits);
    this.applyWorkflow();
  }

  destroy() {
    this.player.stop(false);
    cancelAnimationFrame(this.raf);
    this.player.ctx?.close();
    this.listeners.clear();
  }
}

// --------------------------------------------------------------------------------------------------------------
// Drawing: the waveform tracks (large view) and the strips of the node

// item's source span drawn across [x0, x1]; only the columns in [clip0, clip1] are painted
function drawWaveform(ctx, take, item, x0, x1, top, height, color, clip0 = x0, clip1 = x1) {
  const peaks = take.peaks;
  const buckets = peaks.length / 2;
  const span = item.src[1] - item.src[0];
  const mid = top + height / 2;
  ctx.fillStyle = color;
  const width = Math.max(1, x1 - x0);
  for (let x = Math.max(0, Math.floor(Math.max(x0, clip0))); x < Math.ceil(Math.min(x1, clip1)); x++) {
    const a = item.src[0] + ((x - x0) / width) * span;
    const b = item.src[0] + ((x + 1 - x0) / width) * span;
    let lo = 0, hi = 0;
    for (let i = Math.floor(a / PEAK_STEP); i <= Math.min(buckets - 1, Math.floor(b / PEAK_STEP)); i++) {
      if (peaks[2 * i] < lo) lo = peaks[2 * i];
      if (peaks[2 * i + 1] > hi) hi = peaks[2 * i + 1];
    }
    const y0 = mid - hi * height * 0.48, y1 = mid - lo * height * 0.48;
    ctx.fillRect(x, y0, 1, Math.max(1, y1 - y0));
  }
}

function drawOverview(canvas, view) {
  const rect = canvas.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  const w = Math.max(1, Math.round(rect.width * dpr)), h = Math.max(1, Math.round(rect.height * dpr));
  if (canvas.width !== w || canvas.height !== h) { canvas.width = w; canvas.height = h; }
  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, w, h);
  const layout = view.layout;
  if (!layout) return;
  const scale = w / Math.max(layout.duration, 0.001);
  const speakers = view.take.speakers;
  const laneH = h / speakers.length;
  speakers.forEach((speaker, i) => {
    ctx.fillStyle = i % 2 ? "#121217" : "#141419";
    ctx.fillRect(0, i * laneH, w, laneH);
  });
  const spans = new Map();
  for (const item of layout.items) {
    if (!item.src) continue;
    const span = spans.get(item.line) ?? { start: item.start, end: item.end, speaker: item.speaker };
    span.start = Math.min(span.start, item.start);
    span.end = Math.max(span.end, item.end);
    spans.set(item.line, span);
  }
  for (const span of spans.values()) {
    const lane = Math.max(0, speakers.indexOf(span.speaker));
    ctx.fillStyle = speakerColor(span.speaker) + "22";
    ctx.fillRect(span.start * scale + dpr, lane * laneH + dpr, Math.max(1, (span.end - span.start) * scale - 2 * dpr), laneH - 2 * dpr);
  }
  for (const item of layout.items) {
    if (!item.src) continue;
    const lane = speakers.indexOf(item.speaker);
    drawWaveform(ctx, view.take, item, item.start * scale, item.end * scale, Math.max(0, lane) * laneH, laneH, speakerColor(item.speaker));
  }
  ctx.font = `${Math.round(10 * dpr)}px system-ui`;
  speakers.forEach((speaker, i) => {
    const label = view.voiceLabel(speaker);
    ctx.fillStyle = "rgba(15,15,18,.75)";
    ctx.fillRect(0, i * laneH, ctx.measureText(label).width + 8 * dpr, 14 * dpr);
    ctx.fillStyle = speakerColor(speaker);
    ctx.fillText(label, 4 * dpr, i * laneH + 10.5 * dpr);
  });
  ctx.fillStyle = "#ff5252";
  ctx.fillRect(Math.round(view.player.position() * scale), 0, Math.max(1, dpr), h);
}

// --------------------------------------------------------------------------------------------------------------
// Captions: a reel of the take's lines that glides to the line being spoken, with up to three lines before and after it
// where there is room; the current line follows the voice word by word when an aligner timed it.

class Captions {
  constructor(view, big = false) {
    this.view = view;
    this.reel = el("div", { class: "d3e-reel" });
    this.root = el("div", { class: `d3e-cap${big ? " d3e-cap-big" : ""}` }, this.reel);
    this.rows = [];
    this.index = null;
    this.builtFor = null;
    new ResizeObserver(() => this.place(false)).observe(this.root);
  }

  build() {
    const v = this.view;
    this.rows = v.take.lines.map((line, i) => {
      const words = [];
      const row = el("div", { class: "d3e-cap-row", title: "Click to go to this line" },
        el("span", { class: "d3e-voice", style: `color:${speakerColor(line.speaker)}` }, v.voiceLabel(line.speaker)));
      for (const token of line.tokens) {
        if (!token.spoken) { row.append(el("span", { class: "d3e-cue" }, token.text), " "); continue; }
        const span = el("span", { class: "d3e-word" }, token.text);
        span.addEventListener("click", (e) => {
          if (i !== this.index) return;
          e.stopPropagation();
          const at = v.layout.toEdited(token.start ?? line.start);
          if (at !== null) v.seek(at);
        });
        words.push({ span, token });
        row.append(span, " ");
      }
      row.addEventListener("click", () => {
        const at = v.layout.toEdited(line.start);
        if (at !== null) v.seek(at);
      });
      return { row, words };
    });
    this.reel.replaceChildren(...this.rows.map(({ row }) => row));
    this.root.classList.toggle("d3e-untimed", !v.take.analysis.aligned);
    this.index = null;
  }

  // centre the current line; without `animate` the reel jumps there (a new take, a resize)
  place(animate = true) {
    const current = this.rows[this.index]?.row;
    if (!current) return;
    this.reel.style.transition = animate ? "" : "none";
    const y = this.root.clientHeight / 2 - (current.offsetTop + current.offsetHeight / 2);
    this.reel.style.transform = `translateY(${Math.round(y)}px)`;
    if (!animate) {
      void this.reel.offsetHeight;
      this.reel.style.transition = "";
    }
  }

  update() {
    const v = this.view;
    if (!v.layout || !v.take.lines.length) return;
    if (this.builtFor !== v.layout) {
      this.builtFor = v.layout;
      this.build();
    }
    const source = v.layout.toSource(v.player.position());
    let index = 0;
    v.take.lines.forEach((line, i) => { if (source >= line.start - 1e-3) index = i; });
    if (index !== this.index) {
      const animate = this.index !== null;
      this.index = index;
      this.rows.forEach(({ row }, i) => { row.className = `d3e-cap-row d3e-d${Math.min(Math.abs(i - index), 4)}`; });
      this.place(animate);
    }
    const cuts = v.original ? [] : v.edits.cuts;
    for (const { span, token } of this.rows[index].words) {
      if (token.start === undefined) continue;
      span.classList.toggle("d3e-now", source >= token.start && source < token.end);
      span.classList.toggle("d3e-said", source >= token.end);
      span.classList.toggle("d3e-cut", cuts.some(([a, b]) => token.start >= a - 0.02 && token.end <= b + 0.02));
    }
  }
}

// --------------------------------------------------------------------------------------------------------------
// The large view: an inset window over ComfyUI

class EditorWindow {
  static RULER = 24;
  static LANE = 64;
  static ONE_TRACK = 90;
  static PAD = 8;

  // one lane per voice, or a single lane holding every voice's clips
  lanes() {
    return this.view.oneTrack ? [null] : this.view.take.speakers;
  }

  laneHeight() {
    return this.view.oneTrack ? EditorWindow.ONE_TRACK : EditorWindow.LANE;
  }

  // each line as a clip: its audio pieces, from the first to the last, in edited time
  clips() {
    const byLine = new Map();
    for (const item of this.view.layout.items) {
      if (!item.src) continue;
      const clip = byLine.get(item.line);
      if (clip) {
        clip.start = Math.min(clip.start, item.start);
        clip.end = Math.max(clip.end, item.end);
        clip.items.push(item);
      } else {
        byLine.set(item.line, { line: item.line, speaker: item.speaker, start: item.start, end: item.end, items: [item] });
      }
    }
    return [...byLine.values()];
  }

  constructor(view, onClose) {
    this.view = view;
    this.onClose = onClose;
    this.drag = null;
    this.helpOpen = false;
    this.customOpen = false;
    this.window = el("div", { class: "d3e-window", tabindex: "0" });
    this.root = el("div", { class: "d3e-overlay" }, this.window);
    this.root.addEventListener("mousedown", (e) => { if (e.target === this.root) this.close(); });
    this.build();
    this.unsubscribe = view.subscribe(() => this.update());
    this.window.addEventListener("keydown", (e) => this.onKey(e));
    // narrow windows put Things to check under the tracks, as SceneCut's viewer does below 900 px
    this.resize = new ResizeObserver(() => {
      this.window.classList.toggle("d3e-narrow", this.window.clientWidth < 1100);
      this.drawTimeline();
    });
    this.resize.observe(this.window);
    document.body.append(this.root);
    this.window.focus();
    this.renderAll();
  }

  build() {
    const v = this.view;
    const button = (label, title, onclick, cls = "") =>
      el("button", { class: `d3e-btn ${cls}`, title, onclick: (e) => { onclick(e); this.window.focus(); } }, label);
    this.undoBtn = button("↶ Undo", "Undo the last change (Ctrl+Z)", () => v.undo());
    this.redoBtn = button("↷ Redo", "Redo (Ctrl+Shift+Z)", () => v.redo());
    const bar = el("div", { class: "d3e-bar" },
      el("span", { class: "d3e-title" }, "Drama 3 Editor"),
      el("span", { class: "d3e-dim" }, "Tidy the silences, check each line, cut what you don't want."),
      el("span", { class: "d3e-spacer" }), this.undoBtn, this.redoBtn,
      button("? Help", "Keyboard shortcuts and what the colours mean (?)", () => this.toggleHelp()),
      button(v.applyText.label, v.applyText.title, () => { v.apply(); this.toast(v.applyText.done); }, "d3e-primary"),
      button("✕ Close", "Close the editor; your edits stay on the node (Esc)", () => this.close()));
    this.presets = el("div", { class: "d3e-presets" });
    this.captions = new Captions(v, true);
    this.heads = el("div", { class: "d3e-heads" });
    this.canvas = el("canvas");
    this.scroll = el("div", { class: "d3e-scroll" }, this.canvas);
    this.playBtn = button("▶ Play", "Play or pause (Space)", () => v.toggle(), "d3e-primary");
    this.timeLabel = el("span", { class: "d3e-time" });
    this.afterBtn = button("After", "Hear and see the take with your edits (A switches)", () => v.setOriginal(false));
    this.beforeBtn = button("Before", "Hear and see the take as it was generated (A switches)", () => v.setOriginal(true));
    this.loopBtn = button("⟲ Loop", "Play the selection over and over (L)", () => { v.loop = !v.loop; this.update(); });
    this.cutBtn = button("✂ Cut selection", "Remove the selected part: drag across the tracks to select (Delete)", () => v.cutSelection());
    this.silenceBtn = button("+ Add silence", "Add 0.5 s of silence at the playhead, or at the start of the selection (I)", () => v.insertSilence(0.5));
    const transport = el("div", { class: "d3e-transport" },
      button("⏮", "Back to the start (Home)", () => v.seek(0)),
      button("◀ Line", "Previous line (↑)", () => v.jumpLine(-1)),
      this.playBtn,
      button("Line ▶", "Next line (↓)", () => v.jumpLine(1)),
      this.timeLabel, el("span", { class: "d3e-sep" }),
      el("span", { class: "d3e-dim" }, "Hear"), this.afterBtn, this.beforeBtn, this.loopBtn, el("span", { class: "d3e-sep" }),
      this.cutBtn, this.silenceBtn, el("span", { class: "d3e-spacer" }),
      el("span", { class: "d3e-dim" }, "Zoom"),
      button("−", "Zoom out (−)", () => this.zoomBy(1 / 1.5)),
      button("Fit", "Show the whole take (0)", () => { v.zoom = null; this.drawTimeline(); }),
      button("+", "Zoom in (+)", () => this.zoomBy(1.5)));
    this.checks = el("div", { class: "d3e-checks" });
    this.foot = el("div", { class: "d3e-foot" });
    this.toastEl = el("div", { class: "d3e-toast" });
    this.window.append(bar, this.presets, el("div", { class: "d3e-captions" }, this.captions.root),
      el("div", { class: "d3e-tracks" }, this.heads, this.scroll), transport, this.checks, this.foot, this.toastEl);
    this.bindTimeline();
  }

  close() {
    this.view.pause();
    this.unsubscribe();
    this.resize.disconnect();
    this.root.remove();
    this.onClose();
  }

  toast(text) {
    this.toastEl.textContent = text;
    this.toastEl.classList.add("d3e-show");
    clearTimeout(this.toastTimer);
    this.toastTimer = setTimeout(() => this.toastEl.classList.remove("d3e-show"), 2400);
  }

  toggleHelp() {
    const v = this.view;
    this.helpOpen = !this.helpOpen;
    this.help?.remove();
    this.help = null;
    if (!this.helpOpen) return;
    const rows = [["Space", "Play or pause (plays the selection when there is one)"], ["↑ ↓", "Previous / next line"],
      ["← →", "Back / forward 1 s (Shift: 5 s)"], ["Home End", "Start / end"], ["Click", "Move the playhead (tracks or a word)"],
      ["Drag", "Select a part on the tracks"], ["Delete", "Cut the selection"], ["I", "Add 0.5 s of silence"], ["L", "Loop the selection"],
      ["A", "Hear after / before your edits"], ["Ctrl+Z", "Undo (Ctrl+Shift+Z redo)"], ["+ − 0", "Zoom in / out / fit"],
      ["Ctrl+wheel", "Zoom at the pointer"], ["Ctrl+Enter", v.applyText.label.replace(/^\W+/, "")], ["Esc", "Clear the selection, then close"]];
    const key = (color, text) => el("div", { class: "d3e-key", style: "display:flex;margin:3px 0" }, el("i", { style: `flex:none;background:${color}` }), text);
    this.help = el("div", { class: "d3e-help" }, el("h4", {}, "Keys"),
      el("table", {}, rows.map(([k, d]) => el("tr", {}, el("td", {}, k), el("td", {}, d)))),
      el("h4", {}, "Colours on the tracks"),
      key("#6d4c1f", "A pause the settings shortened (click its label to leave it as generated)"),
      key("#2e7d4f", "Silence added between lines that ran into each other"),
      key("#455a64", "A pause you left as generated"), key("#ef5350", "Audio removed here"),
      key("rgba(90,130,220,.6)", "Your selection"));
    this.window.append(this.help);
  }

  onKey(e) {
    const v = this.view;
    if (e.target.matches?.("input, textarea")) return;
    const ctrl = e.ctrlKey || e.metaKey;
    let handled = true;
    if (e.key === " ") v.toggle();
    else if (e.key === "Escape") { if (this.helpOpen) this.toggleHelp(); else if (v.selection) { v.selection = null; this.update(); } else this.close(); }
    else if (ctrl && e.key.toLowerCase() === "z") e.shiftKey ? v.redo() : v.undo();
    else if (ctrl && e.key.toLowerCase() === "y") v.redo();
    else if (ctrl && e.key === "Enter") { v.apply(); this.toast(v.applyText.done); }
    else if (e.key === "Delete" || e.key === "Backspace") v.cutSelection();
    else if (e.key === "ArrowUp") v.jumpLine(-1);
    else if (e.key === "ArrowDown") v.jumpLine(1);
    else if (e.key === "ArrowLeft") v.seek(v.player.position() - (e.shiftKey ? 5 : 1));
    else if (e.key === "ArrowRight") v.seek(v.player.position() + (e.shiftKey ? 5 : 1));
    else if (e.key === "Home") v.seek(0);
    else if (e.key === "End") v.seek(v.layout?.duration ?? 0);
    else if (e.key.toLowerCase() === "i" && !ctrl) v.insertSilence(0.5);
    else if (e.key.toLowerCase() === "l" && !ctrl) { v.loop = !v.loop; this.update(); }
    else if (e.key.toLowerCase() === "a" && !ctrl) v.setOriginal(!v.original);
    else if (e.key === "+" || e.key === "=") this.zoomBy(1.5);
    else if (e.key === "-") this.zoomBy(1 / 1.5);
    else if (e.key === "0") { v.zoom = null; this.drawTimeline(); }
    else if (e.key === "?") this.toggleHelp();
    else handled = false;
    if (handled) { e.preventDefault(); e.stopPropagation(); }
  }

  // ---- 1 · silence presets, and every value under Custom
  renderPresets() {
    const v = this.view;
    const e = v.edits;
    const matching = PRESETS.find((p) => p.values && Object.entries(p.values).every(([key, value]) => e[key] === value))?.id;
    const chosen = this.customOpen || !matching ? "custom" : matching;
    const cards = el("div", { class: "d3e-cards" }, PRESETS.map((preset) => el("div", {
      class: `d3e-card ${chosen === preset.id ? "d3e-on" : ""}`,
      onclick: () => {
        this.customOpen = !preset.values;
        if (preset.values) v.change(preset.values);
        else this.renderPresets();
      },
    }, el("b", {}, preset.name), el("span", {}, preset.hint))));
    const shortened = v.plan.pauses.filter((p) => p.action === "shortened").length;
    const added = v.plan.pieces.filter((piece) => piece.gap).length - e.inserts.length;
    this.presets.replaceChildren(...[
      el("div", { class: "d3e-step" }, "1 · How should the silences sound?",
        el("small", {}, `${shortened} pause${shortened === 1 ? "" : "s"} shortened · space added between ${added} pair${added === 1 ? "" : "s"} of lines · ` +
          `${fmt(v.take.analysis.duration)} → ${fmt(v.plan.duration)}`)),
      cards, chosen === "custom" ? this.customSettings() : null].filter(Boolean));
  }

  customSettings() {
    const v = this.view;
    const e = v.edits;
    const number = (key, min, max, step, disabled = false) => el("input", { type: "number", min, max, step, value: e[key], disabled,
      onchange: (ev) => v.change({ [key]: +ev.target.value }) });
    const seconds = (input) => el("span", { class: "d3e-unit" }, input, "s");
    const toggle = (key, label) => el("label", { class: "d3e-toggle" },
      el("input", { type: "checkbox", checked: e[key], onchange: (ev) => v.change({ [key]: ev.target.checked }) }), label);
    const field = (label, control) => el("div", { class: "d3e-field" }, el("span", {}, label), control);
    const group = (title, hint, ...rows) => el("div", { class: "d3e-group" }, el("h5", {}, title), ...rows, el("small", {}, hint));
    return el("div", { class: "d3e-custom" },
      group("Start and end", "Cuts the silence before the first word and after the last, leaving a little.",
        toggle("trim_start", "Trim the start"), field("Keep before the first word", seconds(number("lead", 0, 2, 0.05, !e.trim_start))),
        toggle("trim_end", "Trim the end"), field("Keep after the last word", seconds(number("tail", 0, 3, 0.05, !e.trim_end)))),
      group("Pauses inside a line", "The middle of a long pause goes, so the end of the word and the next breath stay.",
        field("Shorten pauses longer than", seconds(number("pause_longer", 0.1, 60, 0.05))),
        field("down to", seconds(number("pause_to", 0, 60, 0.05)))),
      group("Space between lines", "Lines that run into each other get the minimum; longer gaps are cut to the maximum.",
        field("At least", seconds(number("gap_min", 0, 10, 0.05))),
        field("At most", seconds(number("gap_max", 0, 60, 0.05)))),
      group("Silence and loudness", "Lower counts only true silence as silence; higher also counts breaths and room noise.",
        field("Silence is quieter than", el("span", { class: "d3e-unit" },
          el("input", { type: "range", min: -75, max: -25, step: 1, value: e.threshold_db, onchange: (ev) => v.change({ threshold_db: +ev.target.value }) }),
          `${e.threshold_db} dB`)),
        toggle("normalize", "Make it as loud as possible without clipping")));
  }

  // ---- voices: one head per track beside the tracks, or every voice on the single track's head
  renderHeads() {
    const v = this.view;
    const toggle = el("button", { class: "d3e-btn d3e-mini", title: v.oneTrack ? "Give each voice its own track" : "Show every voice on one track",
      onclick: () => v.setOneTrack(!v.oneTrack) }, v.oneTrack ? "Track per voice" : "One track");
    this.heads.replaceChildren(el("div", { class: "d3e-ruler-gap", style: `height:${EditorWindow.RULER}px` }, toggle));
    const mute = (speaker, s) => el("button", { class: `d3e-btn d3e-mini ${s.mute ? "d3e-on" : ""}`,
      title: s.mute ? "Hear this voice again" : "Silence this voice", onclick: () => v.setSpeaker(speaker, { mute: !s.mute }) }, s.mute ? "Muted" : "Mute");
    const name = (speaker, s) => {
      const label = el("span", { class: "d3e-name", title: "Click to name this voice" },
        s.name || `Voice ${speaker}`, s.name ? el("small", {}, `(Voice ${speaker})`) : null);
      label.addEventListener("click", () => this.rename(speaker, label));
      return label;
    };
    if (v.oneTrack) {
      this.heads.append(el("div", { class: "d3e-head", style: `height:${EditorWindow.ONE_TRACK}px` },
        el("span", { class: "d3e-dim" }, "All voices"),
        el("div", { class: "d3e-chips" }, v.take.speakers.map((speaker) => {
          const s = v.edits.speakers[String(speaker)] || {};
          return el("span", { class: "d3e-chip" }, el("span", { class: "d3e-swatch", style: `background:${speakerColor(speaker)}` }), name(speaker, s), mute(speaker, s));
        }))));
      return;
    }
    for (const speaker of v.take.speakers) {
      const s = v.edits.speakers[String(speaker)] || {};
      const gain = s.gain_db || 0;
      this.heads.append(el("div", { class: "d3e-head", style: `height:${EditorWindow.LANE}px` },
        el("div", { class: "d3e-row" }, el("span", { class: "d3e-swatch", style: `background:${speakerColor(speaker)}` }), name(speaker, s), mute(speaker, s)),
        el("div", { class: "d3e-row", title: "This voice's volume" }, el("span", { class: "d3e-dim" }, "Vol"),
          el("input", { type: "range", min: -18, max: 12, step: 0.5, value: gain, onchange: (ev) => v.setSpeaker(speaker, { gain_db: +ev.target.value }) }),
          el("span", { class: "d3e-dim", style: "width:58px;text-align:right;white-space:nowrap" }, `${gain >= 0 ? "+" : ""}${gain.toFixed(1)} dB`))));
    }
  }

  rename(speaker, nameEl) {
    const input = el("input", { type: "text", value: (this.view.edits.speakers[String(speaker)] || {}).name || "", placeholder: `Voice ${speaker}` });
    let done = false;
    const commit = (save) => {
      if (done) return;
      done = true;
      if (save) this.view.setSpeaker(speaker, { name: input.value.trim() });
      else this.renderHeads();
      this.window.focus();
    };
    input.addEventListener("keydown", (e) => {
      if (e.key === "Enter") commit(true);
      else if (e.key === "Escape") commit(false);
      e.stopPropagation();
    });
    input.addEventListener("blur", () => commit(true));
    nameEl.replaceWith(input);
    input.focus();
    input.select();
  }

  // ---- 2 · things to check: lines whose pace looks wrong, long silences, and your own edits
  findings() {
    const v = this.view;
    const lines = v.take.lines;
    const items = [];
    const lineName = (i) => `Line ${i + 1} (${v.voiceLabel(lines[i].speaker)})`;
    lines.forEach((line, i) => {
      const spoken = line.tokens.filter((t) => t.spoken).length;
      const length = line.end - line.start;
      const item = { at: line.start, until: Math.max(line.end, line.start + 2) };
      if (length < 0.05) items.push({ ...item, text: `${lineName(i)} has no audio of its own`, hint: "The take ran it into the line before: listen there." });
      else if (spoken && length / spoken < 0.12) items.push({ ...item, text: `${lineName(i)} is short for its words`, hint: "A word may be missing." });
      else if (spoken && length / spoken > 1.4) items.push({ ...item, text: `${lineName(i)} is long for its words`, hint: "It may hold a long silence or a stall." });
    });
    for (const pause of v.plan.pauses) {
      const length = pause.end - pause.start;
      if (length < LONG_SILENCE || (pause.kind !== "inside" && pause.kind !== "between")) continue;
      const i = lines.findIndex((line) => pause.start < line.end);
      const where = pause.kind === "inside" && i >= 0 ? `inside line ${i + 1}` : `between lines ${i} and ${i + 1}`;
      const kept = pause.action === "kept by you";
      items.push({
        at: Math.max(0, pause.start - 1), until: pause.end + 1, pause, silence: true,
        text: `${length.toFixed(1)} s of silence ${where}`,
        hint: kept ? "Left as generated." : pause.action === "kept" ? "Your settings keep it." : `Shortened to ${pause.to.toFixed(2)} s.`,
      });
    }
    return items.sort((a, b) => a.at - b.at);
  }

  hear(from, until) {
    const v = this.view;
    const start = v.layout.toEdited(from) ?? v.layout.toEdited(Math.max(0, from - 0.5));
    if (start === null) return;
    const end = v.layout.toEdited(Math.max(from, until - 0.01));
    v.play(start, end !== null && end > start ? end : null);
  }

  renderChecks() {
    const v = this.view;
    const e = v.edits;
    const items = this.findings();
    const row = (iconText, iconClass, text, hint, actions, onclick) => el("div", { class: "d3e-item", onclick },
      el("span", { class: iconClass }, iconText), el("div", {}, text, el("small", {}, hint)), el("div", { class: "d3e-acts" }, actions));
    const play = (from, until) => el("button", { class: "d3e-btn d3e-mini", title: "Hear this part",
      onclick: (ev) => { ev.stopPropagation(); this.hear(from, until); } }, "▶ Hear");
    const go = (at) => () => { const t = v.layout.toEdited(at); if (t !== null) v.seek(t, { keepPlaying: false }); };
    const edits = [
      ...e.cuts.map(([a, b], i) => row("✂", "d3e-dim", `${(b - a).toFixed(2)} s cut at ${fmt(a)}`, "Removed by you.",
        [el("button", { class: "d3e-btn d3e-mini", onclick: (ev) => { ev.stopPropagation(); v.removeEdit("cuts", i); } }, "Undo this one")], go(Math.max(0, a - 0.5)))),
      ...e.inserts.map(([at, s], i) => row("+", "d3e-dim", `${s.toFixed(2)} s of silence at ${fmt(at)}`, "Added by you.",
        [el("button", { class: "d3e-btn d3e-mini", onclick: (ev) => { ev.stopPropagation(); v.removeEdit("inserts", i); } }, "Undo this one")], go(at))),
    ];
    this.checks.replaceChildren(...[
      el("div", { class: "d3e-step" }, `2 · Things to check (${items.length})`),
      items.length ? items.map((item) => row(item.silence ? "⏸" : "⚠", "d3e-warn", item.text, item.hint, [
        play(item.at, item.until),
        item.pause && item.pause.action !== "kept" ? el("button", { class: "d3e-btn d3e-mini",
          title: item.pause.action === "kept by you" ? "Shorten it again with your settings" : "Undo this change and keep the silence as generated",
          onclick: (ev) => { ev.stopPropagation(); v.toggleKeep(item.pause.id); } },
        item.pause.action === "kept by you" ? "Shorten it" : "Leave as generated") : null,
      ], go(item.at)))
        : el("div", { class: "d3e-empty" }, `Nothing to check: every line has a normal pace and no silence lasts ${LONG_SILENCE} s or more.`),
      el("div", { class: "d3e-step", style: "margin-top:14px" }, `Your edits (${edits.length})`),
      edits.length ? edits : el("div", { class: "d3e-empty" }, "Drag across the tracks to select a part, then Cut selection or Add silence."),
    ].flat());
  }

  // ---- timeline geometry
  pxPerSecond() {
    const width = this.scroll.clientWidth - 2 * EditorWindow.PAD;
    const fit = width / Math.max(this.view.layout?.duration || 1, 0.5);
    return this.view.zoom ?? fit;
  }

  zoomBy(factor, anchorX = null) {
    const before = this.pxPerSecond();
    const anchor = anchorX ?? this.scroll.clientWidth / 2;
    const t = (this.scroll.scrollLeft + anchor - EditorWindow.PAD) / before;
    this.view.zoom = Math.min(2000, Math.max(5, before * factor));
    this.drawTimeline();
    this.scroll.scrollLeft = t * this.view.zoom + EditorWindow.PAD - anchor;
  }

  timeAt(clientX) {
    const rect = this.canvas.getBoundingClientRect();
    return Math.max(0, (clientX - rect.left - EditorWindow.PAD) / this.pxPerSecond());
  }

  bindTimeline() {
    this.canvas.addEventListener("mousedown", (e) => {
      if (e.button !== 0) return;
      const t = this.timeAt(e.clientX);
      const rect = this.canvas.getBoundingClientRect();
      const hit = this.hitPause(e.clientX - rect.left, e.clientY - rect.top);
      if (hit) { this.view.toggleKeep(hit.id); return; }
      this.drag = { start: t, moved: false };
      const move = (ev) => {
        const now = this.timeAt(ev.clientX);
        if (Math.abs(now - this.drag.start) * this.pxPerSecond() > 3) this.drag.moved = true;
        if (this.drag.moved) {
          this.view.selection = [Math.min(this.drag.start, now), Math.min(Math.max(this.drag.start, now), this.view.layout.duration)];
          this.update();
        }
      };
      const up = () => {
        window.removeEventListener("mousemove", move);
        window.removeEventListener("mouseup", up);
        if (!this.drag.moved) { this.view.selection = null; this.view.seek(this.drag.start); }
        else this.view.seek(this.view.selection[0], { keepPlaying: false });
        this.drag = null;
      };
      window.addEventListener("mousemove", move);
      window.addEventListener("mouseup", up);
    });
    this.scroll.addEventListener("wheel", (e) => {
      if (e.ctrlKey || e.metaKey) {
        e.preventDefault();
        this.zoomBy(e.deltaY < 0 ? 1.25 : 0.8, e.clientX - this.scroll.getBoundingClientRect().left);
      } else if (Math.abs(e.deltaY) > Math.abs(e.deltaX)) {
        e.preventDefault();
        this.scroll.scrollLeft += e.deltaY;
      }
    }, { passive: false });
    this.scroll.addEventListener("scroll", () => this.drawTimeline());
    this.canvas.addEventListener("mousemove", (e) => {
      const rect = this.canvas.getBoundingClientRect();
      const x = e.clientX - rect.left;
      const hit = this.hitPause(x, e.clientY - rect.top);
      const cut = hit ? null : this.cutsShown?.find((c) => Math.abs(c.x - x) <= 4);
      this.canvas.style.cursor = hit ? "pointer" : "text";
      this.canvas.title = hit ? this.describePause(hit) : cut ? `${cut.removed.toFixed(2)} s removed here` : "Click to move the playhead, drag to select";
    });
  }

  describePause(pause) {
    const kind = pause.kind === "inside" ? "Pause inside a line" : "Space between lines";
    const length = (pause.end - pause.start).toFixed(2);
    if (pause.action === "kept by you") return `${kind}: left as generated (${length} s). Click to apply your settings again.`;
    return `${kind}: was ${length} s, now ${pause.to.toFixed(2)} s. Click to leave it as generated.`;
  }

  // pause labels sit on the ruler
  hitPause(x, y) {
    if (y > EditorWindow.RULER || !this.badges) return null;
    return this.badges.find((b) => x >= b.x0 && x <= b.x1)?.pause ?? null;
  }

  drawTimeline() {
    const v = this.view;
    if (!v.layout) return;
    const { RULER, PAD } = EditorWindow;
    const LANE = this.laneHeight();
    const pps = this.pxPerSecond();
    const speakers = v.take.speakers;
    const lanes = this.lanes();
    const width = v.zoom === null ? this.scroll.clientWidth : Math.ceil(2 * PAD + v.layout.duration * pps);
    const height = RULER + lanes.length * LANE;
    const dpr = window.devicePixelRatio || 1;
    const viewWidth = Math.max(width, this.scroll.clientWidth);
    this.canvas.style.width = `${viewWidth}px`;
    this.canvas.style.height = `${height}px`;
    this.scroll.style.height = `${height + (width > this.scroll.clientWidth ? 12 : 0)}px`;
    if (this.canvas.width !== Math.round(viewWidth * dpr) || this.canvas.height !== Math.round(height * dpr)) {
      this.canvas.width = Math.round(viewWidth * dpr);
      this.canvas.height = Math.round(height * dpr);
    }
    const ctx = this.canvas.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, viewWidth, height);
    const left = this.scroll.scrollLeft, right = left + this.scroll.clientWidth;
    const X = (t) => PAD + t * pps;
    const laneTop = (i) => RULER + i * LANE;

    lanes.forEach((_, i) => {
      ctx.fillStyle = i % 2 ? "#121217" : "#141419";
      ctx.fillRect(0, laneTop(i), viewWidth, LANE);
      ctx.fillStyle = "#202027";
      ctx.fillRect(0, laneTop(i) + LANE - 1, viewWidth, 1);
    });
    if (v.selection) {
      ctx.fillStyle = "rgba(90,130,220,.22)";
      ctx.fillRect(X(v.selection[0]), RULER, (v.selection[1] - v.selection[0]) * pps, height - RULER);
    }
    for (const item of v.layout.items) {
      if (!item.gap) continue;
      const x0 = X(item.start), x1 = X(item.end);
      if (x1 < left - 4 || x0 > right + 4) continue;
      ctx.fillStyle = "rgba(120,200,140,.10)";
      ctx.fillRect(x0, RULER, x1 - x0, height - RULER);
      ctx.fillStyle = "#7fc69a";
      ctx.font = "10px system-ui";
      if (x1 - x0 > 34) ctx.fillText(`+${item.gap.toFixed(2)} s`, x0 + 3, height - 5);
    }
    // each line as its own clip: a tinted block with the voice and the words on top and its waveform below
    const position = v.player.position();
    for (const clip of this.clips()) {
      const x0 = X(clip.start), x1 = X(clip.end);
      if (x1 < left - 4 || x0 > right + 4) continue;
      const lane = v.oneTrack ? 0 : Math.max(0, speakers.indexOf(clip.speaker));
      const top = laneTop(lane) + 3, h = LANE - 7;
      const color = speakerColor(clip.speaker);
      const muted = (v.edits.speakers[String(clip.speaker)] || {}).mute;
      const current = position >= clip.start && position < clip.end;
      ctx.beginPath();
      ctx.roundRect(x0 + 0.5, top + 0.5, Math.max(1, x1 - x0 - 1), h, 4);
      ctx.fillStyle = color + (current ? "33" : "1f");
      ctx.fill();
      ctx.lineWidth = current ? 2 : 1;
      ctx.strokeStyle = color + (current ? "ff" : "88");
      ctx.stroke();
      ctx.save();
      ctx.beginPath();
      ctx.rect(x0 + 2, top, Math.max(0, x1 - x0 - 4), h);
      ctx.clip();
      ctx.fillStyle = color + "40";
      ctx.fillRect(x0, top, x1 - x0, 15);
      ctx.fillStyle = "#f0f0f0";
      ctx.font = "11px system-ui";
      ctx.fillText(`${v.voiceLabel(clip.speaker)} · ${v.take.texts[clip.line] ?? ""}`, x0 + 5, top + 11);
      ctx.restore();
      for (const item of clip.items) {
        drawWaveform(ctx, v.take, item, X(item.start), X(item.end), top + 17, h - 19, muted ? "#55555f" : color, Math.max(left - 2, x0 + 1), Math.min(right + 2, x1 - 1));
      }
    }
    // where audio was removed
    this.cutsShown = [];
    for (const cut of v.layout.cuts()) {
      const x = X(cut.at);
      this.cutsShown.push({ x, removed: cut.removed });
      ctx.fillStyle = "#ef5350";
      ctx.fillRect(x - 1, RULER, 2, height - RULER);
      ctx.beginPath();
      ctx.moveTo(x - 5, RULER);
      ctx.lineTo(x + 5, RULER);
      ctx.lineTo(x, RULER + 7);
      ctx.fill();
    }
    // ruler
    ctx.fillStyle = "#18181d";
    ctx.fillRect(0, 0, viewWidth, RULER);
    ctx.fillStyle = "#26262d";
    ctx.fillRect(0, RULER - 1, viewWidth, 1);
    const steps = [0.1, 0.25, 0.5, 1, 2, 5, 10, 15, 30, 60];
    const major = steps.find((s) => s * pps >= 70) ?? 60;
    ctx.fillStyle = "#8d95a6";
    ctx.font = "10px system-ui";
    for (let t = Math.floor(Math.max(0, (left - PAD) / pps) / major) * major; X(t) < right + 50; t += major) {
      ctx.fillRect(X(t), RULER - 6, 1, 6);
      ctx.fillText(major < 1 ? t.toFixed(2) : fmt(t, false), X(t) + 3, 13);
    }
    // what the settings did to each pause: a label where there is room, a dot where it would cover the one before
    this.badges = [];
    if (!v.original) {
      let lastEnd = -Infinity;
      const dots = [];
      for (const pause of v.plan.pauses) {
        if (pause.action === "kept" || pause.kind === "lead" || pause.kind === "tail") continue;
        const at = v.layout.toEdited(Math.max(pause.start, 0));
        if (at === null) continue;
        const length = pause.end - pause.start;
        const label = pause.action === "kept by you" ? "as generated" : `${length.toFixed(2)} → ${pause.to.toFixed(2)} s`;
        const color = pause.action === "kept by you" ? "#455a64" : pause.action === "lengthened" ? "#2e7d4f" : "#6d4c1f";
        const w = ctx.measureText(label).width + 8;
        if (X(at) < lastEnd + 2) {
          dots.push({ x: X(at), color, pause });
          continue;
        }
        ctx.fillStyle = color;
        ctx.fillRect(X(at), 3, w, 15);
        ctx.fillStyle = "#fff";
        ctx.fillText(label, X(at) + 4, 14);
        this.badges.push({ x0: X(at), x1: X(at) + w, pause });
        lastEnd = X(at) + w;
      }
      for (const dot of dots) {
        ctx.beginPath();
        ctx.arc(dot.x + 5, 10, 5, 0, Math.PI * 2);
        ctx.fillStyle = dot.color;
        ctx.fill();
        ctx.lineWidth = 1.5;
        ctx.strokeStyle = "#0f0f12";
        ctx.stroke();
        this.badges.unshift({ x0: dot.x, x1: dot.x + 10, pause: dot.pause });
      }
    }
    const p = X(v.player.position());
    ctx.fillStyle = "#ff5252";
    ctx.fillRect(p - 1, 0, 2, height);
  }

  renderFoot() {
    const v = this.view;
    const take = v.take;
    const before = take.analysis.duration, after = v.plan.duration;
    const key = (color, text) => el("span", { class: "d3e-key" }, el("i", { style: `background:${color}` }), text);
    const sel = v.selection;
    this.foot.replaceChildren(...[
      el("span", {}, "Take ", el("b", {}, fmt(before)), " → with your edits ", el("b", {}, fmt(after)),
        ` (${after <= before ? "−" : "+"}${Math.abs(before - after).toFixed(1)} s) · ${take.lines.length} lines · ${take.speakers.length} voice${take.speakers.length > 1 ? "s" : ""}`),
      sel ? el("span", {}, el("b", {}, `Selected ${fmt(sel[0])} – ${fmt(sel[1])}`), " · Delete cuts it, I adds silence, L loops it")
        : el("span", {}, v.original ? "You are hearing the take as generated: press After for your edits." : "Click the tracks to move the playhead, drag to select."),
      el("span", { class: "d3e-spacer" }),
      take.analysis.aligned ? null : el("span", { class: "d3e-warn" }, "Connect Qwen3-ForcedAligner to the aligner input to follow each word."),
      key("#6d4c1f", "shortened"), key("#2e7d4f", "space added"), key("#ef5350", "removed")].filter(Boolean));
  }

  renderAll() {
    if (!this.view.layout) return;
    this.layoutShown = null;
    this.update();
  }

  update() {
    const v = this.view;
    if (!v.layout) return;
    if (this.planShown !== v.plan || this.editsShown !== v.edits || this.layoutShown !== v.layout || this.oneTrackShown !== v.oneTrack) {
      this.planShown = v.plan;
      this.editsShown = v.edits;
      this.layoutShown = v.layout;
      this.oneTrackShown = v.oneTrack;
      this.renderPresets();
      this.renderHeads();
      this.renderChecks();
    }
    this.playBtn.textContent = v.player.playing ? "⏸ Pause" : "▶ Play";
    this.timeLabel.textContent = `${fmt(v.player.position())} / ${fmt(v.layout.duration)}`;
    this.afterBtn.classList.toggle("d3e-on", !v.original);
    this.beforeBtn.classList.toggle("d3e-on", v.original);
    this.undoBtn.disabled = !v.undoStack.length;
    this.redoBtn.disabled = !v.redoStack.length;
    this.loopBtn.classList.toggle("d3e-on", v.loop);
    this.cutBtn.disabled = !v.selection || v.original;
    this.silenceBtn.disabled = v.original;
    if (this.selectionShown !== v.selection || this.originalShown !== v.original || this.footFor !== v.plan) {
      this.selectionShown = v.selection;
      this.originalShown = v.original;
      this.footFor = v.plan;
      this.renderFoot();
    }
    this.drawTimeline();
    this.captions.update();
    if (v.player.playing) this.follow();
  }

  // keep the playhead in view while playing
  follow() {
    const x = EditorWindow.PAD + this.view.player.position() * this.pxPerSecond();
    const left = this.scroll.scrollLeft, width = this.scroll.clientWidth;
    if (x > left + width - 60 || x < left) this.scroll.scrollLeft = x - 40;
  }
}

// --------------------------------------------------------------------------------------------------------------
// The compact preview (the node, the app's take): captions, one strip per voice, play controls and the Edit button

class TakePreview {
  constructor(view, { onOpen, emptyText }) {
    this.view = view;
    this.captions = new Captions(view);
    this.canvas = el("canvas", { title: "Click to move the playhead · double-click to open the editor" });
    this.playBtn = el("button", { class: "d3e-btn d3e-primary", title: "Play or pause", onclick: () => view.toggle() }, "▶");
    this.time = el("span", { class: "d3e-time" });
    this.status = el("div", { class: "d3e-status" }, emptyText);
    this.root = el("div", { class: "d3e-node" }, this.captions.root, this.canvas,
      el("div", { class: "d3e-row" },
        el("button", { class: "d3e-btn", title: "Back to the start", onclick: () => view.seek(0) }, "⏮"), this.playBtn, this.time,
        el("span", { class: "d3e-grow" }),
        el("button", { class: "d3e-btn", title: "Open the editor: tidy the silences, check each line, cut and name voices",
          onclick: () => view.layout && onOpen() }, icon(EXPAND_ICON), "Edit")),
      this.status);
    this.canvas.addEventListener("dblclick", () => view.layout && onOpen());
    this.canvas.addEventListener("click", (e) => {
      if (!view.layout) return;
      const rect = this.canvas.getBoundingClientRect();
      view.seek(((e.clientX - rect.left) / rect.width) * view.layout.duration);
    });
    this.unsubscribe = view.subscribe(() => this.update());
    new ResizeObserver(() => drawOverview(this.canvas, view)).observe(this.canvas);
  }

  setStatus(text) {
    this.status.textContent = text ?? "";
    this.status.style.display = text ? "flex" : "none";
  }

  update() {
    const view = this.view;
    if (!view.layout) return;
    this.playBtn.textContent = view.player.playing ? "⏸" : "▶";
    this.time.textContent = `${fmt(view.player.position())} / ${fmt(view.layout.duration)}`;
    this.captions.update();
    drawOverview(this.canvas, view);
  }
}

// the take as the Python side wrote it: its analysis (JSON) and the unedited audio
async function loadTake(view, analysisUrl, audioUrl) {
  const [analysisRes, audioRes] = await Promise.all([fetch(analysisUrl), fetch(audioUrl)]);
  if (!analysisRes.ok || !audioRes.ok) throw new Error(`take not found (HTTP ${analysisRes.status}/${audioRes.status})`);
  const analysis = await analysisRes.json();
  const buffer = await view.player.context().decodeAudioData(await audioRes.arrayBuffer());
  await view.load(analysis, buffer);
}

export { injectStyle, EditorView, EditorWindow, TakePreview, loadTake };
