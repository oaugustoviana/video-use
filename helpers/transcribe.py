"""Transcribe a video locally with WhisperX (free, runs on this machine).

Extracts mono 16kHz audio via ffmpeg, runs WhisperX transcription plus
word-level alignment, optionally runs speaker diarization (pyannote, needs a
Hugging Face token in HF_TOKEN), and writes the result to
<edit_dir>/transcripts/<video_stem>.json using the same flat {"words": [...]}
schema the rest of video-use (pack_transcripts.py etc.) already expects.

Cached: if the output file already exists, transcription is skipped.

Usage:
    python helpers/transcribe.py <video_path>
    python helpers/transcribe.py <video_path> --edit-dir /custom/edit
    python helpers/transcribe.py <video_path> --language pt
    python helpers/transcribe.py <video_path> --num-speakers 2
    python helpers/transcribe.py <video_path> --model medium --device cuda
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8")

DEFAULT_MODEL = "small"
DEFAULT_LANGUAGE = "pt"

_model_cache: dict = {}


def load_api_key() -> str | None:
    """Load an optional Hugging Face token, used only for speaker diarization.

    Returns None if absent. Transcription still works fine without it, it just
    will not separate "who said what" when there is more than one speaker.
    """
    for candidate in [Path(__file__).resolve().parent.parent / ".env", Path(".env")]:
        if candidate.exists():
            for line in candidate.read_text().splitlines():
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                if k.strip() in ("HF_TOKEN", "HUGGINGFACE_TOKEN"):
                    val = v.strip().strip('"').strip("'")
                    if val:
                        return val
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN") or None


def extract_audio(video_path: Path, dest: Path) -> None:
    cmd = [
        "ffmpeg", "-y", "-i", str(video_path),
        "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le",
        str(dest),
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def _default_device_and_compute() -> tuple[str, str]:
    import torch
    if torch.cuda.is_available():
        return "cuda", "float16"
    return "cpu", "int8"


def _load_whisper_model(model_name: str, device: str, compute_type: str, language: str | None):
    key = (model_name, device, compute_type, language)
    if key not in _model_cache:
        import whisperx
        _model_cache[key] = whisperx.load_model(
            model_name, device, compute_type=compute_type, language=language
        )
    return _model_cache[key]


def run_whisperx(
    audio_path: Path,
    model_name: str = DEFAULT_MODEL,
    device: str | None = None,
    language: str | None = DEFAULT_LANGUAGE,
    num_speakers: int | None = None,
    hf_token: str | None = None,
) -> dict:
    import whisperx

    if device is None:
        device, compute_type = _default_device_and_compute()
    else:
        compute_type = "float16" if device == "cuda" else "int8"

    model = _load_whisper_model(model_name, device, compute_type, language)
    audio = whisperx.load_audio(str(audio_path))
    result = model.transcribe(audio, batch_size=8)

    align_model, metadata = whisperx.load_align_model(
        language_code=result.get("language", language or "pt"), device=device
    )
    result = whisperx.align(
        result["segments"], align_model, metadata, audio, device, return_char_alignments=False
    )
    del align_model
    gc.collect()

    if hf_token:
        try:
            diarize_model = whisperx.diarize.DiarizationPipeline(token=hf_token, device=device)
            diarize_kwargs = {}
            if num_speakers:
                diarize_kwargs["min_speakers"] = num_speakers
                diarize_kwargs["max_speakers"] = num_speakers
            diarize_segments = diarize_model(audio, **diarize_kwargs)
            result = whisperx.assign_word_speakers(diarize_segments, result)
        except Exception as e:
            # Diarization is a nice-to-have (who's speaking). Never let it take down
            # the transcript itself, e.g. if a gated HF model needs its terms accepted.
            print(f"  diarization skipped (transcript still saved): {e}", flush=True)

    return result


def to_scribe_schema(whisperx_result: dict) -> dict:
    """Flatten WhisperX segments/words into the flat {"words": [...]} schema
    pack_transcripts.py and the rest of video-use expect."""
    words: list[dict] = []
    for seg in whisperx_result.get("segments", []):
        for w in seg.get("words", []):
            start = w.get("start")
            end = w.get("end")
            if start is None or end is None:
                continue
            entry = {
                "type": "word",
                "text": w.get("word", ""),
                "start": start,
                "end": end,
            }
            speaker = w.get("speaker")
            if speaker:
                entry["speaker_id"] = speaker
            words.append(entry)
    return {"words": words}


def transcribe_one(
    video: Path,
    edit_dir: Path,
    api_key: str | None = None,
    language: str | None = None,
    num_speakers: int | None = None,
    verbose: bool = True,
    model: str = DEFAULT_MODEL,
    device: str | None = None,
) -> Path:
    """Transcribe a single video locally with WhisperX. Returns path to transcript JSON.

    Cached: returns existing path immediately if the transcript already exists.
    `api_key` here is an optional Hugging Face token, used only for diarization.
    """
    transcripts_dir = edit_dir / "transcripts"
    transcripts_dir.mkdir(parents=True, exist_ok=True)
    out_path = transcripts_dir / f"{video.stem}.json"

    if out_path.exists():
        if verbose:
            print(f"cached: {out_path.name}")
        return out_path

    if verbose:
        print(f"  extracting audio from {video.name}", flush=True)

    t0 = time.time()
    with tempfile.TemporaryDirectory() as tmp:
        audio = Path(tmp) / f"{video.stem}.wav"
        extract_audio(video, audio)
        if verbose:
            print(f"  transcribing locally with WhisperX ({model})", flush=True)
        result = run_whisperx(
            audio,
            model_name=model,
            device=device,
            language=language or DEFAULT_LANGUAGE,
            num_speakers=num_speakers,
            hf_token=api_key,
        )

    payload = to_scribe_schema(result)
    out_path.write_text(json.dumps(payload, indent=2))
    dt = time.time() - t0

    if verbose:
        kb = out_path.stat().st_size / 1024
        print(f"  saved: {out_path.name} ({kb:.1f} KB) in {dt:.1f}s")
        print(f"    words: {len(payload['words'])}")

    return out_path


def main() -> None:
    ap = argparse.ArgumentParser(description="Transcribe a video locally with WhisperX")
    ap.add_argument("video", type=Path, help="Path to video file")
    ap.add_argument(
        "--edit-dir",
        type=Path,
        default=None,
        help="Edit output directory (default: <video_parent>/edit)",
    )
    ap.add_argument(
        "--language",
        type=str,
        default=DEFAULT_LANGUAGE,
        help="ISO language code (default: pt). Pass empty string to auto-detect.",
    )
    ap.add_argument(
        "--num-speakers",
        type=int,
        default=None,
        help="Optional number of speakers when known. Improves diarization accuracy.",
    )
    ap.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help="WhisperX model size (default: small). Options: tiny, base, small, medium, large-v2, large-v3.",
    )
    ap.add_argument(
        "--device",
        type=str,
        default=None,
        help="Force device: cuda or cpu. Default: auto-detect (this machine has a 2GB GPU, cpu is the safe default for larger models).",
    )
    args = ap.parse_args()

    video = args.video.resolve()
    if not video.exists():
        sys.exit(f"video not found: {video}")

    edit_dir = (args.edit_dir or (video.parent / "edit")).resolve()
    hf_token = load_api_key()

    transcribe_one(
        video=video,
        edit_dir=edit_dir,
        api_key=hf_token,
        language=args.language or None,
        num_speakers=args.num_speakers,
        model=args.model,
        device=args.device,
    )


if __name__ == "__main__":
    main()
