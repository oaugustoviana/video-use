#!/usr/bin/env python3
"""Export a video-use edl.json into a gapless FCP7 XML (xmeml v5) timeline.

Imports cleanly into Adobe Premiere Pro (File > Import) and DaVinci Resolve
(File > Import > Timeline > AAF/XML). The point is a NON-DESTRUCTIVE handoff:
the cut decisions arrive on the timeline referencing the ORIGINAL source files,
so the editor keeps every trim, ripple and roll available in Premiere.

Adapted from Vic Laranja's perfect-cuts export_fcp7.py (single-source) and
generalized to the video-use edl.json schema, which supports MULTIPLE sources.

Usage:
    python helpers/export_premiere.py edit/edl.json -o "entrevista corte.xml"
    python helpers/export_premiere.py edit/edl.json -o out.xml --name "Corte 1"

edl.json (video-use schema — times in SECONDS):
{
  "sources": {"C0103": "/abs/C0103.MP4", "C0108": "/abs/C0108.MP4"},
  "ranges": [
    {"source": "C0103", "start": 2.42, "end": 6.85, ...},
    {"source": "C0108", "start": 14.30, "end": 28.90, ...}
  ]
}

Ranges are laid back-to-back from frame 0 — zero gap space by construction.
Each range keeps its own source in/out, so multi-camera edits survive the round trip.
"""
import argparse
import json
import math
import os
import subprocess
import urllib.parse
import uuid

# NTSC fractional rates map to their integer timebase with ntsc=TRUE.
NTSC = {23.976: 24, 24000 / 1001: 24, 29.97: 30, 30000 / 1001: 30,
        59.94: 60, 60000 / 1001: 60}


def rate_info(fps):
    for ntsc_fps, tb in NTSC.items():
        if abs(fps - ntsc_fps) < 0.01:
            return tb, "TRUE"
    return round(fps), "FALSE"


def ffprobe_meta(path):
    """Read fps, width, height, samplerate, duration from a media file."""
    def probe(*args):
        out = subprocess.run(
            ["ffprobe", "-v", "error", *args, "-of",
             "default=noprint_wrappers=1:nokey=1", path],
            capture_output=True, text=True)
        return out.stdout.strip().splitlines()

    # Query each field separately: ffprobe emits fields in its own internal
    # order, not the order requested, so a combined query mis-pairs the values.
    rfr = (probe("-select_streams", "v:0", "-show_entries", "stream=r_frame_rate") or ["30/1"])[0]
    num, den = (rfr.split("/") + ["1"])[:2]
    fps = float(num) / float(den) if float(den) else float(num)
    wv = probe("-select_streams", "v:0", "-show_entries", "stream=width")
    hv = probe("-select_streams", "v:0", "-show_entries", "stream=height")
    width = int(wv[0]) if wv else 1920
    height = int(hv[0]) if hv else 1080

    # Phones/mirrorless store landscape pixels + a rotation flag. Premiere shows
    # the DISPLAY orientation, so swap W/H on a 90/270 rotation to match.
    rot = 0
    for entry in ("stream_side_data=rotation", "stream_tags=rotate"):
        r = probe("-select_streams", "v:0", "-show_entries", entry)
        if r:
            try:
                rot = int(float(r[0]))
                break
            except ValueError:
                pass
    if abs(rot) % 180 == 90:
        width, height = height, width

    a = probe("-select_streams", "a:0", "-show_entries", "stream=sample_rate")
    samplerate = int(a[0]) if a and a[0].isdigit() else 48000

    ac = probe("-select_streams", "a:0", "-show_entries", "stream=channels")
    channels = int(ac[0]) if ac and ac[0].isdigit() else 2

    d = probe("-show_entries", "format=duration")
    duration = float(d[0]) if d and d[0].replace(".", "").isdigit() else 0.0

    return {"fps": fps, "width": width, "height": height,
            "samplerate": samplerate, "channels": channels, "duration": duration}


def file_block(file_id, src_name, pathurl, tb, ntsc, src_frames, w, h, sr, ch, video):
    """Full <file> definition — emitted the FIRST time a source appears."""
    video_chars = f"""      <video>
        <samplecharacteristics>
          <rate><timebase>{tb}</timebase><ntsc>{ntsc}</ntsc></rate>
          <width>{w}</width>
          <height>{h}</height>
          <anamorphic>FALSE</anamorphic>
          <pixelaspectratio>square</pixelaspectratio>
          <fielddominance>none</fielddominance>
        </samplecharacteristics>
      </video>
""" if video else ""
    return f"""    <file id="{file_id}">
      <name>{src_name}</name>
      <pathurl>{pathurl}</pathurl>
      <rate><timebase>{tb}</timebase><ntsc>{ntsc}</ntsc></rate>
      <duration>{src_frames}</duration>
      <media>
{video_chars}      <audio>
        <samplecharacteristics>
          <samplerate>{sr}</samplerate>
          <sampledepth>16</sampledepth>
        </samplecharacteristics>
        <channelcount>{ch}</channelcount>
      </audio>
      </media>
    </file>"""


def links(n):
    return f"""            <link>
              <linkclipref>clipitem-video-{n}</linkclipref>
              <mediatype>video</mediatype>
              <trackindex>1</trackindex>
              <clipindex>{n}</clipindex>
            </link>
            <link>
              <linkclipref>clipitem-audio-{n}</linkclipref>
              <mediatype>audio</mediatype>
              <trackindex>1</trackindex>
              <clipindex>{n}</clipindex>
              <groupindex>1</groupindex>
            </link>"""


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("edl", help="path to video-use edl.json")
    ap.add_argument("-o", "--output", required=True, help="output .xml path")
    ap.add_argument("--name", help="sequence name (default: derived from output)")
    args = ap.parse_args()

    edl = json.load(open(args.edl, encoding="utf-8"))
    sources = edl["sources"]
    ranges = edl["ranges"]
    if not ranges:
        raise SystemExit("edl has no ranges — nothing to export")

    # Probe every source once; cache the metadata.
    meta = {sid: ffprobe_meta(path) for sid, path in sources.items()}

    # Sequence rate = the fps of the first range's source. Warn on mismatch.
    seq_fps = meta[ranges[0]["source"]]["fps"]
    tb, ntsc = rate_info(seq_fps)
    seq_w = meta[ranges[0]["source"]]["width"]
    seq_h = meta[ranges[0]["source"]]["height"]
    seq_sr = meta[ranges[0]["source"]]["samplerate"]
    for sid, m in meta.items():
        if abs(m["fps"] - seq_fps) > 0.01:
            print(f"WARNING: source {sid} is {m['fps']:.3f}fps but sequence is "
                  f"{seq_fps:.3f}fps — Premiere will conform it; verify sync.")

    # Assign a stable file id per source, in first-appearance order.
    file_ids, seen = {}, []
    for r in ranges:
        if r["source"] not in file_ids:
            file_ids[r["source"]] = f"file-{len(file_ids) + 1}"

    vitems, aitems, timeline = [], [], 0
    for n, r in enumerate(ranges, 1):
        sid = r["source"]
        m = meta[sid]
        fps = m["fps"]
        in_frame = int(math.floor(r["start"] * fps))
        out_frame = int(math.ceil(r["end"] * fps))
        dur = out_frame - in_frame
        if dur <= 0:
            print(f"WARNING: range {n} ({sid}) has non-positive length — skipped.")
            continue
        start, end = timeline, timeline + dur
        timeline = end

        fid = file_ids[sid]
        src_path = sources[sid]
        src_name = os.path.basename(src_path)
        stem = os.path.splitext(src_name)[0]
        pathurl = "file://" + urllib.parse.quote(src_path.replace("\\", "/"))
        src_frames = int(math.ceil(m["duration"] * fps)) or out_frame

        common = f"""            <name>{stem}</name>
            <enabled>TRUE</enabled>
            <duration>{dur}</duration>
            <start>{start}</start>
            <end>{end}</end>
            <in>{in_frame}</in>
            <out>{out_frame}</out>"""

        # Reemit the full <file> block every time (perfect-cuts behavior, proven
        # to import into Premiere). Premiere de-duplicates by file id + pathurl,
        # so repeated definitions of the same source are safe.
        vfile = file_block(fid, src_name, pathurl, tb, ntsc, src_frames,
                           m["width"], m["height"], m["samplerate"], m["channels"], True)
        afile = file_block(fid, src_name, pathurl, tb, ntsc, src_frames,
                           m["width"], m["height"], m["samplerate"], m["channels"], False)

        vitems.append(f"""          <clipitem id="clipitem-video-{n}">
{common}
{vfile}
            <sourcetrack>
              <mediatype>video</mediatype>
              <trackindex>1</trackindex>
            </sourcetrack>
{links(n)}
          </clipitem>""")
        aitems.append(f"""          <clipitem id="clipitem-audio-{n}">
{common}
{afile}
            <sourcetrack>
              <mediatype>audio</mediatype>
              <trackindex>1</trackindex>
            </sourcetrack>
            <channelcount>{m["channels"]}</channelcount>
{links(n)}
          </clipitem>""")

    # Overlay track (V2): keyword-motion / graphic clips placed at OUTPUT times.
    # Each overlay is its own file (a mov/png with alpha), video-only, and the
    # track may have gaps (overlays only appear on their key moments). Read from
    # edl["overlays"]: [{"file": ..., "start_in_output": s, "duration": s}, ...].
    oitems = []
    edl_dir = os.path.dirname(os.path.abspath(args.edl))
    for n, ov in enumerate(edl.get("overlays", []), 1):
        ov_path = ov["file"]
        if not os.path.isabs(ov_path):
            ov_path = os.path.normpath(os.path.join(edl_dir, ov_path))
        start_f = int(round(ov["start_in_output"] * seq_fps))
        dur_f = int(round(ov["duration"] * seq_fps))
        if dur_f <= 0:
            print(f"WARNING: overlay {n} has non-positive duration — skipped.")
            continue
        end_f = start_f + dur_f
        ov_name = os.path.basename(ov_path)
        ov_stem = os.path.splitext(ov_name)[0]
        ov_url = "file://" + urllib.parse.quote(ov_path.replace("\\", "/"))
        fid = f"file-ov-{n}"
        oitems.append(f"""          <clipitem id="clipitem-ov-{n}">
            <name>{ov_stem}</name>
            <enabled>TRUE</enabled>
            <duration>{dur_f}</duration>
            <start>{start_f}</start>
            <end>{end_f}</end>
            <in>0</in>
            <out>{dur_f}</out>
            <compositemode>normal</compositemode>
            <file id="{fid}">
              <name>{ov_name}</name>
              <pathurl>{ov_url}</pathurl>
              <rate><timebase>{tb}</timebase><ntsc>{ntsc}</ntsc></rate>
              <duration>{dur_f}</duration>
              <media>
                <video>
                  <samplecharacteristics>
                    <rate><timebase>{tb}</timebase><ntsc>{ntsc}</ntsc></rate>
                    <width>{seq_w}</width>
                    <height>{seq_h}</height>
                    <anamorphic>FALSE</anamorphic>
                    <pixelaspectratio>square</pixelaspectratio>
                    <fielddominance>none</fielddominance>
                  </samplecharacteristics>
                </video>
              </media>
            </file>
            <sourcetrack>
              <mediatype>video</mediatype>
              <trackindex>1</trackindex>
            </sourcetrack>
          </clipitem>""")
    overlay_track = f"""        <track>
{chr(10).join(oitems)}
        </track>
""" if oitems else ""

    seq_name = args.name or os.path.splitext(os.path.basename(args.output))[0]
    uid = str(uuid.uuid4())
    xml = f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE xmeml>
<xmeml version="5">
  <sequence id="sequence-{uid}">
    <uuid>{uid}</uuid>
    <name>{seq_name}</name>
    <duration>{timeline}</duration>
    <rate><timebase>{tb}</timebase><ntsc>{ntsc}</ntsc></rate>
    <in>0</in>
    <out>{timeline}</out>
    <timecode>
      <rate><timebase>{tb}</timebase><ntsc>{ntsc}</ntsc></rate>
      <frame>0</frame>
      <displayformat>NDF</displayformat>
    </timecode>
    <media>
      <video>
        <format>
          <samplecharacteristics>
            <rate><timebase>{tb}</timebase><ntsc>{ntsc}</ntsc></rate>
            <width>{seq_w}</width>
            <height>{seq_h}</height>
            <anamorphic>FALSE</anamorphic>
            <pixelaspectratio>square</pixelaspectratio>
            <fielddominance>none</fielddominance>
          </samplecharacteristics>
        </format>
        <track>
{chr(10).join(vitems)}
        </track>
{overlay_track}      </video>
      <audio>
        <numOutputChannels>2</numOutputChannels>
        <format>
          <samplecharacteristics>
            <samplerate>{seq_sr}</samplerate>
            <sampledepth>16</sampledepth>
          </samplecharacteristics>
        </format>
        <track>
{chr(10).join(aitems)}
        </track>
      </audio>
    </media>
  </sequence>
</xmeml>
"""
    with open(args.output, "w", encoding="utf-8") as f:
        f.write(xml)
    secs = timeline / seq_fps
    print(f"{len(ranges)} clips from {len(file_ids)} source(s), {secs:.2f}s -> {args.output}")


if __name__ == "__main__":
    main()
