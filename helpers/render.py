"""Render a video from an EDL.

Implements the HEURISTICS render pipeline in the correct order:

  1. Per-segment extract with color grade + 30ms audio fades baked in
  2. Lossless -c copy concat into base.mp4
  3. If overlays or subtitles: single filter graph that overlays animations
     (with PTS shift so frame 0 lands at the overlay window start)
     and applies `subtitles` filter LAST → final.mp4

Optionally builds a master SRT from the per-source transcripts + EDL
output-timeline offsets, applies the proven force_style (2-word
UPPERCASE chunks, Helvetica 18 Bold, MarginV=35).

Usage:
    python helpers/render.py <edl.json> -o final.mp4
    python helpers/render.py <edl.json> -o preview.mp4 --preview
    python helpers/render.py <edl.json> -o final.mp4 --build-subtitles
    python helpers/render.py <edl.json> -o final.mp4 --no-subtitles
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

# Windows consoles often default stdout to a legacy code page (cp1252) that
# can't encode characters like the arrows used in this file's log messages
# or accented transcript text. Force UTF-8 so prints don't crash mid-render.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8")

try:
    from grade import get_preset, auto_grade_for_clip  # same directory
except Exception:
    def get_preset(name: str) -> str:
        return ""

    def auto_grade_for_clip(video, start=0.0, duration=None, verbose=False):  # type: ignore
        return "eq=contrast=1.03:saturation=0.98", {}


# -------- Subtitle style (bold-overlay, proven at 1920×1080 and 1080×1920) --
#
# MarginV is NOT taste — it is a platform safe-zone rule.
# TikTok / IG Reels / Shorts UI (caption, username, music, right-rail actions)
# covers roughly the bottom ~25–30% of a 1080×1920 frame. Captions placed near
# the bottom edge get clipped or obscured by the UI. libass auto-scales the
# render canvas relative to PlayResY=288, so MarginV=90 lands the caption
# baseline roughly 30% up from the bottom on any aspect — clear of the UI on
# every major vertical-video platform. Do not drop this below ~75 without a
# specific reason.
SUB_FORCE_STYLE = (
    "FontName=Helvetica,FontSize=18,Bold=1,"
    "PrimaryColour=&H00FFFFFF,OutlineColour=&H00000000,BackColour=&H00000000,"
    "BorderStyle=1,Outline=2,Shadow=0,"
    "Alignment=2,MarginV=90"
)

# -------- Helpers ------------------------------------------------------------


def run(cmd: list[str], quiet: bool = False) -> None:
    if not quiet:
        print(f"  $ {' '.join(str(c) for c in cmd[:6])}{' …' if len(cmd) > 6 else ''}")
    subprocess.run(cmd, check=True)


def resolve_grade_filter(grade_field: str | None) -> str:
    """The EDL's 'grade' field can be a preset name, a raw ffmpeg filter, or 'auto'.

    Returns the filter string to embed into the per-segment -vf chain.
    For 'auto', returns the sentinel "__AUTO__" which is resolved per-segment.
    """
    if not grade_field:
        return ""
    if grade_field == "auto":
        return "__AUTO__"
    # Preset names are short identifiers, filter strings contain '=' or ','.
    if re.fullmatch(r"[a-zA-Z0-9_\-]+", grade_field):
        try:
            return get_preset(grade_field)
        except KeyError:
            print(f"warning: unknown preset '{grade_field}', using as raw filter")
            return grade_field
    return grade_field


def resolve_path(maybe_path: str, base: Path) -> Path:
    """Resolve a path that may be absolute or relative to `base`."""
    p = Path(maybe_path)
    if p.is_absolute():
        return p
    return (base / p).resolve()


# -------- HDR → SDR tone mapping (HLG / PQ sources) --------------------------
#
# iPhone defaults to HLG HDR in Rec.2020 (and many mirrorless cameras ship PQ).
# If the source is HDR and we only downconvert bit depth (yuv420p10le → yuv420p)
# without tone-mapping, the output is 8-bit but still carries HLG/PQ transfer
# metadata. Players that honor the metadata (screen recorders, most social
# upload re-encodes) interpret 8-bit values in an HDR container and the result
# looks oversaturated / blown out. QuickTime on macOS can hide this locally —
# screen recording and uploaded renders cannot.
#
# Fix: detect HDR via color_transfer and prepend a zscale+tonemap chain to the
# vf graph so the output is clean Rec.709 SDR.

HDR_TRANSFERS = {"smpte2084", "arib-std-b67"}  # PQ (HDR10) and HLG

TONEMAP_CHAIN = (
    "zscale=t=linear:npl=100,"
    "format=gbrpf32le,"
    "zscale=p=bt709,"
    "tonemap=tonemap=hable:desat=0,"
    "zscale=t=bt709:m=bt709:r=tv,"
    "format=yuv420p"
)


def is_hdr_source(video: Path) -> bool:
    """Return True if the source uses a PQ or HLG transfer function."""
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=color_transfer",
             "-of", "default=noprint_wrappers=1:nokey=1", str(video)],
            capture_output=True, text=True, check=True,
        )
        return out.stdout.strip() in HDR_TRANSFERS
    except subprocess.CalledProcessError:
        return False


def get_video_dims(video: Path) -> tuple[int, int]:
    """Return (width, height) as DISPLAYED (post-rotation) — swaps coded dims
    when a +/-90 degree rotation side-data tag is present, same detection
    `is_portrait_source` uses below."""
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height",
         "-of", "csv=p=0", str(video)],
        capture_output=True, text=True, check=True,
    )
    w_str, h_str = [p for p in out.stdout.strip().split(",") if p][:2]
    w, h = int(w_str), int(h_str)

    rot_out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream_side_data=rotation",
         "-of", "csv=p=0", str(video)],
        capture_output=True, text=True, check=True,
    )
    rotation_text = rot_out.stdout.strip().strip(",")
    if rotation_text:
        try:
            rotation = int(float(rotation_text.splitlines()[0]))
            if rotation % 180 != 0:
                w, h = h, w
        except ValueError:
            pass
    return w, h


def is_portrait_source(video: Path) -> bool:
    """Return True if the video displays portrait / vertical.

    Checks coded width/height, then flips the verdict if a +/-90 degree
    rotation side-data (Display Matrix) is present — common on phone/gimbal
    footage that stores landscape-coded frames with a rotation tag so the
    player displays them upright as portrait.
    """
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=width,height",
             "-of", "csv=p=0", str(video)],
            capture_output=True, text=True, check=True,
        )
        w_str, h_str = [p for p in out.stdout.strip().split(",") if p][:2]
        w, h = int(w_str), int(h_str)
        portrait = h > w

        rot_out = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream_side_data=rotation",
             "-of", "csv=p=0", str(video)],
            capture_output=True, text=True, check=True,
        )
        rotation_text = rot_out.stdout.strip().strip(",")
        if rotation_text:
            try:
                rotation = int(float(rotation_text.splitlines()[0]))
                if rotation % 180 != 0:
                    portrait = not portrait
            except ValueError:
                pass
        return portrait
    except Exception:
        return False


# -------- Ken Burns push in/out ----------------------------------------------
#
# Static tripod talking-head shots read as flat/amateur on Reels compared to a
# slow continuous push. `zoompan` applied to a VIDEO input (not a still) needs
# d=1 (consume exactly one input frame per output frame — the default d=125
# is for holding a single photo) and must be driven by 'on' (the running
# output-frame counter) rather than a per-call increment, else the zoom level
# resets every filter re-init. x/y re-center on every frame so the crop stays
# frame-centered as zoom grows.


def build_zoom_filter(
    direction: str,
    duration: float,
    fps: int,
    out_w: int,
    out_h: int,
    z_from: float = 1.0,
    z_to: float = 1.10,
) -> str:
    """Build a zoompan filter string for a slow push in/out over `duration`.

    direction: "in" (z_from -> z_to, e.g. 1.0 -> 1.10) or "out" (reversed).
    out_w/out_h must match the frame size AFTER the preceding scale filter —
    zoompan needs an explicit target size, it does not inherit input dims.
    """
    total_frames = max(1, round(duration * fps))
    if direction == "out":
        z_from, z_to = z_to, z_from
    step = (z_to - z_from) / total_frames
    z_expr = f"if(lte(on,1),{z_from:.5f},min(zoom+{step:.6f},{z_to:.5f}))" if direction != "out" \
        else f"if(lte(on,1),{z_from:.5f},max(zoom-{abs(step):.6f},{z_to:.5f}))"
    return (
        f"zoompan=z='{z_expr}':d=1:"
        f"x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':"
        f"s={out_w}x{out_h}:fps={fps}"
    )


# -------- Per-segment extraction (Rule 2 + Rule 3) --------------------------


def extract_segment(
    source: Path,
    seg_start: float,
    duration: float,
    grade_filter: str,
    out_path: Path,
    preview: bool = False,
    draft: bool = False,
    voice_chain: bool = False,
    zoom: dict | None = None,
) -> None:
    """Extract a cut range as its own MP4 with grade + 30ms audio fades baked in.

    `-ss` before `-i` for fast accurate seeking. Scale to 1080p from 4K.
    Portrait sources (height > width) are scaled by height to preserve orientation.

    Quality ladder:
      - final (default): 1080p libx264 fast CRF 20
      - preview:         1080p libx264 medium CRF 22 (evaluable for QC)
      - draft:           720p libx264 ultrafast CRF 28 (cut-point check only)

    `voice_chain=True` runs a per-clip mic-polish pass (highpass -> gentle
    multiband-ish compression -> light denoise) before the edge fades, for
    takes whose recorded level is inconsistent within themselves (see
    metodologia/playbooks/edicao-video.md item 3).

    `zoom={"direction": "in"|"out", "from": 1.0, "to": 1.10}` bakes a slow
    Ken Burns push into the segment (see build_zoom_filter) — static tripod
    talking-head shots read as flat without it. Omit for B-roll/handheld
    shots that already carry their own camera movement.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)

    portrait = is_portrait_source(source)
    target_dim = 1280 if draft else 1920
    if portrait:
        scale = f"scale=-2:{target_dim}"
    else:
        scale = f"scale={target_dim}:-2"

    vf_parts: list[str] = []
    if is_hdr_source(source):
        vf_parts.append(TONEMAP_CHAIN)
    vf_parts.append(scale)
    if grade_filter:
        vf_parts.append(grade_filter)
    if zoom:
        src_w, src_h = get_video_dims(source)
        if portrait:
            out_w = round((src_w * target_dim / src_h) / 2) * 2
            out_h = target_dim
        else:
            out_w = target_dim
            out_h = round((src_h * target_dim / src_w) / 2) * 2
        vf_parts.append(build_zoom_filter(
            zoom.get("direction", "in"), duration, fps=24,
            out_w=out_w, out_h=out_h,
            z_from=float(zoom.get("from", 1.0)), z_to=float(zoom.get("to", 1.10)),
        ))
    vf = ",".join(vf_parts)

    # 30ms audio fades at both edges (Rule 3) — prevent pops
    fade_out_start = max(0.0, duration - 0.03)
    fade_parts = [f"afade=t=in:st=0:d=0.03", f"afade=t=out:st={fade_out_start:.3f}:d=0.03"]

    if voice_chain:
        # Real chain, replicated from the Premiere reference (lapela mic):
        # Multiband Compressor (Broadcast preset) -> Parametric EQ (Vocal
        # Enhancer) -> DeNoise. See metodologia/playbooks/edicao-video.md #3.
        # ffmpeg has no single "multiband compressor" filter -- acrossover
        # splits into the same 4 bands (137/1147/6910Hz crossovers) as the
        # Premiere preset, each band gets its own acompressor, then amix
        # recombines and alimiter mirrors the reference's -5dB final limiter.
        eq_chain = (
            "highpass=f=80,"
            "equalizer=f=110:width_type=q:width=1:g=6,"
            "equalizer=f=291:width_type=q:width=1:g=-3,"
            "treble=g=14:f=17500:width_type=h:width=3000"
        )
        fade_chain = ",".join(fade_parts)
        af_complex = (
            f"[0:a]{eq_chain}[apre];"
            f"[apre]acrossover=split=137 1147 6910[b0][b1][b2][b3];"
            f"[b0]acompressor=threshold=-20dB:ratio=2.5:attack=1:release=100[c0];"
            f"[b1]acompressor=threshold=-21dB:ratio=2.5:attack=1:release=100[c1];"
            f"[b2]acompressor=threshold=-22dB:ratio=2.5:attack=1:release=100[c2];"
            f"[b3]acompressor=threshold=-23dB:ratio=2.5:attack=1:release=100[c3];"
            f"[c0][c1][c2][c3]amix=inputs=4:normalize=0[mb];"
            f"[mb]alimiter=limit=0.562[lim];"
            f"[lim]afftdn=nr=8:nf=-40,{fade_chain}[aout]"
        )
    else:
        af_complex = f"[0:a]{','.join(fade_parts)}[aout]"

    if draft:
        preset, crf = "ultrafast", "28"
    elif preview:
        preset, crf = "medium", "22"
    else:
        preset, crf = "fast", "20"

    filter_complex = f"[0:v]{vf}[vout];{af_complex}"
    cmd = [
        "ffmpeg", "-y",
        "-ss", f"{seg_start:.3f}",
        "-i", str(source),
        "-t", f"{duration:.3f}",
        "-filter_complex", filter_complex,
        "-map", "[vout]", "-map", "[aout]",
        "-c:v", "libx264", "-preset", preset, "-crf", crf,
        "-pix_fmt", "yuv420p", "-r", "24",
        "-c:a", "aac", "-b:a", "192k", "-ar", "48000",
        "-movflags", "+faststart",
        str(out_path),
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)


def _has_voice_chain(r: dict, edl: dict) -> bool:
    if "voice_chain" in r:
        return bool(r["voice_chain"])
    return bool(edl.get("voice_chain"))


def extract_all_segments(
    edl: dict,
    edit_dir: Path,
    preview: bool,
    draft: bool = False,
) -> list[Path]:
    """Extract every EDL range into edit_dir/clips_graded/seg_NN.mp4.
    Returns the ordered list of segment paths.

    If the EDL `grade` is "auto", analyze each segment range with
    `auto_grade_for_clip` and apply a per-segment subtle correction.
    Otherwise, apply the same preset/raw filter to every segment.
    """
    resolved = resolve_grade_filter(edl.get("grade"))
    is_auto = resolved == "__AUTO__"
    clips_dir = edit_dir / (
        "clips_draft" if draft else ("clips_preview" if preview else "clips_graded")
    )
    clips_dir.mkdir(parents=True, exist_ok=True)

    ranges = edl["ranges"]
    sources = edl["sources"]

    seg_paths: list[Path] = []
    print(f"extracting {len(ranges)} segment(s) → {clips_dir.name}/")
    if is_auto:
        print("  (auto-grade per segment: analyzing each range)")
    for i, r in enumerate(ranges):
        src_name = r["source"]
        src_path = resolve_path(sources[src_name], edit_dir)
        start = float(r["start"])
        end = float(r["end"])
        duration = end - start
        out_path = clips_dir / f"seg_{i:02d}_{src_name}.mp4"

        if is_auto:
            seg_filter, _stats = auto_grade_for_clip(src_path, start=start, duration=duration, verbose=False)
        else:
            seg_filter = resolved

        note = r.get("beat") or r.get("note") or ""
        vchain = _has_voice_chain(r, edl)
        zoom = r.get("zoom")
        print(f"  [{i:02d}] {src_name}  {start:7.2f}-{end:7.2f}  ({duration:5.2f}s)  {note}"
              + ("  [voice-chain]" if vchain else "")
              + (f"  [zoom {zoom['direction']} {zoom.get('from',1.0)}->{zoom.get('to',1.10)}]" if zoom else ""))
        if is_auto:
            print(f"        grade: {seg_filter or '(none)'}")
        extract_segment(src_path, start, duration, seg_filter, out_path, preview=preview, draft=draft, voice_chain=vchain, zoom=zoom)
        seg_paths.append(out_path)

    return seg_paths


# -------- Lossless concat ----------------------------------------------------


def concat_segments(segment_paths: list[Path], out_path: Path, edit_dir: Path) -> None:
    """Lossless concat via the concat demuxer. No re-encode."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    concat_list = edit_dir / "_concat.txt"
    concat_list.write_text("".join(f"file '{p.resolve()}'\n" for p in segment_paths), encoding="utf-8")

    cmd = [
        "ffmpeg", "-y",
        "-f", "concat", "-safe", "0",
        "-i", str(concat_list),
        "-c", "copy",
        "-movflags", "+faststart",
        str(out_path),
    ]
    print(f"concat → {out_path.name}")
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    concat_list.unlink(missing_ok=True)


# -------- Master SRT (Rule 5) ------------------------------------------------


PUNCT_BREAK = set(".,!?;:")


def _srt_timestamp(seconds: float) -> str:
    total_ms = int(round(seconds * 1000))
    h, rem = divmod(total_ms, 3600_000)
    m, rem = divmod(rem, 60_000)
    s, ms = divmod(rem, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def _words_in_range(transcript: dict, t_start: float, t_end: float) -> list[dict]:
    out: list[dict] = []
    for w in transcript.get("words", []):
        if w.get("type") != "word":
            continue
        ws = w.get("start")
        we = w.get("end")
        if ws is None or we is None:
            continue
        if we <= t_start or ws >= t_end:
            continue
        out.append(w)
    return out


def build_master_srt(edl: dict, edit_dir: Path, out_path: Path) -> None:
    """Build an output-timeline SRT from per-source transcripts.

    - 2-word chunks (break on any punctuation in between)
    - UPPERCASE text
    - Output times computed as word.start - segment_start + segment_offset
    """
    transcripts_dir = edit_dir / "transcripts"
    sources = edl["sources"]

    entries: list[tuple[float, float, str]] = []
    seg_offset = 0.0

    for r in edl["ranges"]:
        src_name = r["source"]
        seg_start = float(r["start"])
        seg_end = float(r["end"])
        seg_duration = seg_end - seg_start

        tr_path = transcripts_dir / f"{src_name}.json"
        if not tr_path.exists():
            print(f"  no transcript for {src_name}, skipping captions for this segment")
            seg_offset += seg_duration
            continue

        transcript = json.loads(tr_path.read_text(encoding="utf-8"))
        words_in_seg = _words_in_range(transcript, seg_start, seg_end)

        # Group into 2-word chunks, break on punctuation
        chunks: list[list[dict]] = []
        current: list[dict] = []
        for w in words_in_seg:
            text = (w.get("text") or "").strip()
            if not text:
                continue
            current.append(w)
            # Break if the current text ends in punctuation or we hit 2 words
            ends_in_punct = bool(text) and text[-1] in PUNCT_BREAK
            if len(current) >= 2 or ends_in_punct:
                chunks.append(current)
                current = []
        if current:
            chunks.append(current)

        for chunk in chunks:
            local_start = max(seg_start, chunk[0].get("start", seg_start))
            local_end = min(seg_end, chunk[-1].get("end", seg_end))
            out_start = max(0.0, local_start - seg_start) + seg_offset
            out_end = max(0.0, local_end - seg_start) + seg_offset
            if out_end <= out_start:
                out_end = out_start + 0.4
            text = " ".join((w.get("text") or "").strip() for w in chunk)
            text = re.sub(r"\s+", " ", text).strip()
            # Strip trailing punctuation for cleaner uppercase look
            text = text.rstrip(",;:")
            text = text.upper()
            entries.append((out_start, out_end, text))

        seg_offset += seg_duration

    # Sort and write as SRT
    entries.sort(key=lambda e: e[0])
    lines: list[str] = []
    for i, (a, b, t) in enumerate(entries, start=1):
        lines.append(str(i))
        lines.append(f"{_srt_timestamp(a)} --> {_srt_timestamp(b)}")
        lines.append(t)
        lines.append("")
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"master SRT -> {out_path.name} ({len(entries)} cues)")


# -------- Loudness normalization (social-ready audio) -----------------------


# Target: -16 LUFS integrated, -1 dBTP peak, LRA 11 LU.
# House style do Augusto (confirmado 2026-07-20): os cortes dele no Premiere
# entregam -18 a -23 LUFS, mais baixo que o padrão de plataforma (-14). -16 é o
# meio-termo escolhido — sentar dentro do range dele sem ficar baixo demais pra
# conteúdo que pode virar anúncio. Ver metodologia/playbooks/edicao-video.md.
LOUDNORM_I = -16.0
LOUDNORM_TP = -1.0
LOUDNORM_LRA = 11.0


def measure_loudness(video_path: Path) -> dict[str, str] | None:
    """Run ffmpeg loudnorm first pass and parse the JSON measurement.

    Returns a dict with measured_i, measured_tp, measured_lra, measured_thresh,
    target_offset, or None if measurement failed.
    """
    filter_str = (
        f"loudnorm=I={LOUDNORM_I}:TP={LOUDNORM_TP}:LRA={LOUDNORM_LRA}:print_format=json"
    )
    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-nostats",
        "-i", str(video_path),
        "-af", filter_str,
        "-vn", "-f", "null", "-",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    # loudnorm prints the JSON to stderr at the end of the run
    stderr = proc.stderr

    # Find the JSON block — loudnorm output contains a `{ ... }` block
    start = stderr.rfind("{")
    end = stderr.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        data = json.loads(stderr[start : end + 1])
    except json.JSONDecodeError:
        return None
    needed = {"input_i", "input_tp", "input_lra", "input_thresh", "target_offset"}
    if not needed.issubset(data.keys()):
        return None
    return data


def apply_loudnorm_two_pass(
    input_path: Path,
    output_path: Path,
    preview: bool = False,
) -> bool:
    """Run two-pass loudnorm on input_path, write normalized copy to output_path.

    Returns True on success, False if measurement failed (caller should fall
    back to copying the input unchanged).

    In preview mode, skips the measurement pass and uses a one-pass approximation
    for speed. Final mode always does the proper two-pass.
    """
    if preview:
        # One-pass approximation — faster, slightly less accurate.
        filter_str = f"loudnorm=I={LOUDNORM_I}:TP={LOUDNORM_TP}:LRA={LOUDNORM_LRA}"
        cmd = [
            "ffmpeg", "-y", "-hide_banner", "-nostats",
            "-i", str(input_path),
            "-c:v", "copy",
            "-af", filter_str,
            "-c:a", "aac", "-b:a", "192k", "-ar", "48000",
            "-movflags", "+faststart",
            str(output_path),
        ]
        print(f"  loudnorm (1-pass preview) → {output_path.name}")
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        return True

    # Full two-pass
    print(f"  loudnorm pass 1: measuring {input_path.name}")
    measurement = measure_loudness(input_path)
    if measurement is None:
        print("  loudnorm measurement failed — falling back to 1-pass")
        return apply_loudnorm_two_pass(input_path, output_path, preview=True)

    print(f"    measured: I={measurement['input_i']} LUFS  "
          f"TP={measurement['input_tp']}  LRA={measurement['input_lra']}")

    filter_str = (
        f"loudnorm=I={LOUDNORM_I}:TP={LOUDNORM_TP}:LRA={LOUDNORM_LRA}"
        f":measured_I={measurement['input_i']}"
        f":measured_TP={measurement['input_tp']}"
        f":measured_LRA={measurement['input_lra']}"
        f":measured_thresh={measurement['input_thresh']}"
        f":offset={measurement['target_offset']}"
        f":linear=true"
    )
    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-nostats",
        "-i", str(input_path),
        "-c:v", "copy",
        "-af", filter_str,
        "-c:a", "aac", "-b:a", "192k", "-ar", "48000",
        "-movflags", "+faststart",
        str(output_path),
    ]
    print(f"  loudnorm pass 2: normalizing → {output_path.name}")
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    return True


# -------- Final compositing (Rule 1 + Rule 4) -------------------------------


def build_final_composite(
    base_path: Path,
    overlays: list[dict],
    subtitles_path: Path | None,
    out_path: Path,
    edit_dir: Path,
    subtitle_style: str = SUB_FORCE_STYLE,
) -> None:
    """Final pass: base → overlays (PTS-shifted) → subtitles LAST → out.

    If there are no overlays and no subtitles, just copy base to out.
    """
    has_overlays = bool(overlays)
    has_subs = subtitles_path is not None and subtitles_path.exists()

    if not has_overlays and not has_subs:
        # Nothing to do — just rename/copy base to final name
        run(["ffmpeg", "-y", "-i", str(base_path), "-c", "copy", str(out_path)], quiet=True)
        return

    inputs: list[str] = ["-i", str(base_path)]
    for ov in overlays:
        ov_path = resolve_path(ov["file"], edit_dir)
        inputs += ["-i", str(ov_path)]

    filter_parts: list[str] = []
    # PTS-shift every overlay so its frame 0 lands at start_in_output
    for idx, ov in enumerate(overlays, start=1):
        t = float(ov["start_in_output"])
        filter_parts.append(f"[{idx}:v]setpts=PTS-STARTPTS+{t}/TB[a{idx}]")

    # Chain overlays on top of base
    current = "[0:v]"
    for idx, ov in enumerate(overlays, start=1):
        t = float(ov["start_in_output"])
        dur = float(ov["duration"])
        end = t + dur
        next_label = f"[v{idx}]"
        filter_parts.append(
            f"{current}[a{idx}]overlay=enable='between(t,{t:.3f},{end:.3f})'{next_label}"
        )
        current = next_label

    # Subtitles LAST — Rule 1
    if has_subs:
        # ffmpeg's filtergraph parser treats backslash as an escape char, which
        # collides with Windows path separators — normalize to forward slashes
        # before escaping the drive-letter colon and any quotes.
        subs_abs = (
            str(subtitles_path.resolve())
            .replace("\\", "/")
            .replace(":", r"\:")
            .replace("'", r"\'")
        )
        filter_parts.append(
            f"{current}subtitles='{subs_abs}':force_style='{subtitle_style}'[outv]"
        )
        out_label = "[outv]"
    else:
        # Rename the last overlay output to [outv] for consistency
        if has_overlays:
            filter_parts.append(f"{current}null[outv]")
            out_label = "[outv]"
        else:
            out_label = "[0:v]"

    filter_complex = ";".join(filter_parts)

    cmd = [
        "ffmpeg", "-y",
        *inputs,
        "-filter_complex", filter_complex,
        "-map", out_label,
        "-map", "0:a",
        "-c:v", "libx264", "-preset", "fast", "-crf", "18",
        "-pix_fmt", "yuv420p",
        "-c:a", "copy",
        "-movflags", "+faststart",
        str(out_path),
    ]
    print(f"compositing → {out_path.name}")
    print(f"  overlays: {len(overlays)}, subtitles: {'yes' if has_subs else 'no'}")
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)


# -------- Music bed (playbook item 2) ----------------------------------------


def mix_music_bed(
    video_path: Path,
    music_path: Path,
    out_path: Path,
    gain_db: float = -18.0,
    notch_freq: float = 1400.0,
    notch_gain: float = -15.0,
    notch_width: float = 700.0,
) -> None:
    """Mix a constant music bed under the whole video (covers cut gaps too).

    Reference values (metodologia/playbooks/edicao-video.md #2): notch the
    music's EQ in the voice-presence range (~1.4kHz, ~-15dB, ~700Hz wide) so
    it doesn't fight the voice, then drop the track well below the voice
    (-15 to -20dB) rather than auto-ducking. Track choice is always manual —
    `music_path` must be supplied by the caller, never auto-selected.
    """
    dur_out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=nw=1:nk=1", str(video_path)],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    duration = float(dur_out)

    filter_complex = (
        f"[1:a]atrim=0:{duration:.3f},"
        f"equalizer=f={notch_freq}:width_type=h:width={notch_width}:g={notch_gain},"
        f"volume={gain_db}dB[music];"
        f"[0:a][music]amix=inputs=2:duration=first:normalize=0[aout]"
    )
    cmd = [
        "ffmpeg", "-y",
        "-i", str(video_path),
        "-i", str(music_path),
        "-filter_complex", filter_complex,
        "-map", "0:v", "-map", "[aout]",
        "-c:v", "copy",
        "-c:a", "aac", "-b:a", "192k", "-ar", "48000",
        "-movflags", "+faststart",
        str(out_path),
    ]
    print(f"  music bed ({music_path.name}, {gain_db}dB, notch {notch_freq}Hz) → {out_path.name}")
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)


# -------- Main ---------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(description="Render a video from an EDL")
    ap.add_argument("edl", type=Path, help="Path to edl.json")
    ap.add_argument("-o", "--output", type=Path, required=True, help="Output video path")
    ap.add_argument(
        "--preview",
        action="store_true",
        help="Preview mode: 1080p, medium, CRF 22 — evaluable for QC, faster than final.",
    )
    ap.add_argument(
        "--draft",
        action="store_true",
        help="Draft mode: 720p, ultrafast, CRF 28 — cut-point verification only.",
    )
    ap.add_argument(
        "--build-subtitles",
        action="store_true",
        help="Build master.srt from transcripts + EDL offsets before compositing",
    )
    ap.add_argument(
        "--no-subtitles",
        action="store_true",
        help="Skip subtitles even if the EDL references one",
    )
    ap.add_argument(
        "--no-loudnorm",
        action="store_true",
        help="Skip audio loudness normalization. Default is on (-16 LUFS, -1 dBTP, LRA 11).",
    )
    args = ap.parse_args()

    edl_path = args.edl.resolve()
    if not edl_path.exists():
        sys.exit(f"edl not found: {edl_path}")

    edl = json.loads(edl_path.read_text())
    edit_dir = edl_path.parent
    out_path = args.output.resolve()

    # 1. Extract per-segment (auto-grade per range if EDL grade is "auto")
    segment_paths = extract_all_segments(
        edl, edit_dir, preview=args.preview, draft=args.draft
    )

    # 2. Concat → base
    if args.draft:
        base_name = "base_draft.mp4"
    elif args.preview:
        base_name = "base_preview.mp4"
    else:
        base_name = "base.mp4"
    base_path = edit_dir / base_name
    concat_segments(segment_paths, base_path, edit_dir)

    # 3. Subtitles: build if requested, resolve final path
    subs_path: Path | None = None
    if not args.no_subtitles:
        if args.build_subtitles:
            subs_path = edit_dir / "master.srt"
            build_master_srt(edl, edit_dir, subs_path)
        elif edl.get("subtitles"):
            subs_path = resolve_path(edl["subtitles"], edit_dir)
            if not subs_path.exists():
                print(f"warning: subtitles path in EDL does not exist: {subs_path}")
                subs_path = None

    # 4. Composite (overlays + subtitles LAST) → intermediate (pre-loudnorm) path
    overlays = edl.get("overlays") or []
    subtitle_style = edl.get("subtitle_style") or SUB_FORCE_STYLE
    tmp_composite = out_path.with_suffix(".prenorm.mp4")
    build_final_composite(
        base_path, overlays, subs_path, tmp_composite, edit_dir,
        subtitle_style=subtitle_style,
    )

    # 4b. Music bed — mixed in before loudnorm so the final pass measures the
    # true combined (voice + music) loudness, not just the voice.
    music = edl.get("music")
    if music:
        music_path = resolve_path(music["file"], edit_dir)
        if music_path.exists():
            tmp_music = out_path.with_suffix(".premusic.mp4")
            tmp_composite.rename(tmp_music)
            mix_music_bed(
                tmp_music, music_path, tmp_composite,
                gain_db=float(music.get("gain_db", -18.0)),
                notch_freq=float(music.get("notch_freq", 1400.0)),
                notch_gain=float(music.get("notch_gain", -15.0)),
                notch_width=float(music.get("notch_width", 700.0)),
            )
            tmp_music.unlink(missing_ok=True)
        else:
            print(f"warning: music path in EDL does not exist: {music_path}")

    if args.no_loudnorm:
        tmp_composite.rename(out_path)
    else:
        print("loudness normalization → -16 LUFS / -1 dBTP / LRA 11")
        apply_loudnorm_two_pass(tmp_composite, out_path, preview=args.draft)
        tmp_composite.unlink(missing_ok=True)

    size_mb = out_path.stat().st_size / (1024 * 1024)
    print(f"\ndone: {out_path} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
