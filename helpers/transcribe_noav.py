"""Transcreve WAV 16k mono com faster-whisper, driblando o PyAV bloqueado pelo Windows.

O bloqueio é so no decode de audio do av; o ffmpeg ja converteu pra WAV, entao
stubamos o modulo `av` em sys.modules e passamos numpy direto pro modelo.
"""
import sys
import types
import wave
import numpy as np

from importlib.machinery import ModuleSpec

for name in ("av", "av.audio", "av.audio.frame", "av.audio.resampler",
             "av.audio.codeccontext", "av.error", "av.container"):
    mod = types.ModuleType(name)
    mod.__spec__ = ModuleSpec(name, None)
    mod.__version__ = "0.0.0"
    sys.modules[name] = mod
sys.modules["av"].AudioResampler = object
sys.modules["av"].AudioFrame = object
sys.modules["av"].open = lambda *a, **k: None

from faster_whisper import WhisperModel  # noqa: E402


def load_wav(path):
    with wave.open(path, "rb") as w:
        assert w.getsampwidth() == 2 and w.getnchannels() == 1, "esperado 16-bit mono"
        raw = w.readframes(w.getnframes())
    return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0


model = WhisperModel("medium", device="cpu", compute_type="int8")

for path in sys.argv[1:]:
    audio = load_wav(path)
    print(f"\n===== {path}  ({len(audio) / 16000:.1f}s) =====", flush=True)
    segments, _ = model.transcribe(audio, language="pt", beam_size=5,
                                   vad_filter=True, condition_on_previous_text=False)
    for s in segments:
        print(f"[{s.start:6.1f}] {s.text.strip()}", flush=True)
