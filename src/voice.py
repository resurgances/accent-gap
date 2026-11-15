"""
Speak a translation in the speaker's own voice, then CHECK it.

Used by demo.py and src/voice_check.py. Runs in the main .venv; the voice model
itself runs in a separate process (src/voice_worker.py in .venv-tts), see there.

Steps for one translation:
  1. split the text into chunks of whole sentences (the voice model works best on
     short inputs)
  2. for each chunk: generate speech in the reference voice (Chatterbox Multilingual)
  3. accuracy check: Whisper (medium: in a test on Hindi, small misheard far more than
     the voice model mis-spoke) listens to the generated speech in the target language,
     and we compare what it heard with the text it was supposed to say
     (character error rate, CER). If the mismatch is above MAX_CER, generate again
     with a different random seed (up to MAX_TRIES) and keep the best attempt.
  4. glue the chunks together with a short pause.

What the check can and cannot tell you:
  - it catches the voice model skipping, repeating, or garbling words
  - it does NOT check whether the translation itself is correct (that is NLLB's job,
    and we did not measure it), and Whisper can itself mishear, so a mismatch is
    not always the voice model's fault
"""
import json
import os
import re
import subprocess
import tempfile
import threading
import unicodedata

import jiwer
import librosa
import numpy as np
import soundfile as sf

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TTS_PYTHON = os.path.join(ROOT, ".venv-tts", "bin", "python")
WORKER = os.path.join(ROOT, "src", "voice_worker.py")
TMP = os.path.join(tempfile.gettempdir(), "accent_gap_voice")

# Languages the voice model (Chatterbox Multilingual) can speak
SUPPORTED = {"ar", "da", "de", "el", "en", "es", "fi", "fr", "he", "hi", "it", "ja", "ko", "ms", "nl",
             "no", "pl", "pt", "ru", "sv", "sw", "tr", "zh"}
MAX_CER = 0.15      # retry a chunk if more than 15% of its characters don't match
MAX_TRIES = 3       # at most 3 attempts per chunk
CHUNK_CHARS = 180   # roughly 10-15 s of speech per chunk
MIN_REF_SECONDS = 6  # the model reads ~6-10 s of the reference voice

_proc = None
_lock = threading.Lock()


def available():
    return os.path.exists(TTS_PYTHON)


def _worker():
    """Start voice_worker.py once and keep it running (loading the model takes a while)."""
    global _proc
    if _proc is None or _proc.poll() is not None:
        os.makedirs(TMP, exist_ok=True)
        _proc = subprocess.Popen([TTS_PYTHON, WORKER], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                 stderr=None, text=True, bufsize=1)
        ready = json.loads(_proc.stdout.readline())
        assert ready.get("ready"), ready
    return _proc


def _ask(req):
    """Send one request. If the worker has just exited (it restarts itself when its GPU
    memory grows too much), start a new one and try once more."""
    global _proc
    with _lock:
        for _ in range(2):
            p = _worker()
            try:
                p.stdin.write(json.dumps(req) + "\n")
                p.stdin.flush()
                line = p.stdout.readline()
            except BrokenPipeError:
                line = ""
            if line:
                return json.loads(line)
            p.wait()
            _proc = None
        raise RuntimeError("the voice worker stopped unexpectedly")


def compare(wav_a_16k, wav_b_16k):
    """Voice similarity between two real recordings (16 kHz arrays)."""
    os.makedirs(TMP, exist_ok=True)
    pa, pb = os.path.join(TMP, "cmp_a.wav"), os.path.join(TMP, "cmp_b.wav")
    sf.write(pa, wav_a_16k, 16000)
    sf.write(pb, wav_b_16k, 16000)
    return _ask({"compare": [pa, pb]})["voice_similarity"]


def chunks(text):
    """Group whole sentences into chunks of at most ~CHUNK_CHARS characters."""
    sentences = [s for s in re.split(r"(?<=[.!?।؟。！？])\s*", text.strip()) if s.strip()]
    out = []
    for s in sentences:
        if out and len(out[-1]) + len(s) + 1 <= CHUNK_CHARS:
            out[-1] += " " + s
        else:
            out.append(s)
    return out


def _clean(t):
    """Lowercase and drop punctuation/symbols. (Not regex \\w: that would also drop
    Hindi vowel signs, which Unicode counts as 'marks', not letters.)"""
    t = "".join(" " if unicodedata.category(c)[0] in "PS" else c for c in t.lower())
    return re.sub(r"\s+", " ", t).strip()


def cer(target, heard):
    """Character error rate between the text we wanted and what Whisper heard."""
    a, b = _clean(target), _clean(heard)
    if not a:
        return 0.0
    return min(1.0, jiwer.cer(a, b)) if b else 1.0


def speak(ref_audio_16k, text, lang, listen):
    """
    ref_audio_16k: the speaker's real voice (16 kHz numpy array), used as the voice to copy
    text, lang   : what to say and in which language (e.g. "fr")
    listen(x, lang) -> str : a speech-to-text function used for the accuracy check (Whisper)
    Returns (sample_rate, audio, report) where report is a list of dicts, one per chunk.
    """
    os.makedirs(TMP, exist_ok=True)
    ref_path = os.path.join(TMP, "ref.wav")
    sf.write(ref_path, ref_audio_16k, 16000)
    pieces, report, sr = [], [], 24000
    for i, chunk in enumerate(chunks(text)):
        best = None
        for attempt in range(MAX_TRIES):
            out_path = os.path.join(TMP, f"chunk{i}_try{attempt}.wav")
            r = _ask({"ref_wav": ref_path, "text": chunk, "lang": lang, "out_wav": out_path, "seed": attempt})
            if not r["ok"]:
                raise RuntimeError(r["error"])
            wav, sr = sf.read(out_path, dtype="float32")
            heard = listen(librosa.resample(wav, orig_sr=sr, target_sr=16000), lang)
            res = {"text": chunk, "heard": heard, "cer": cer(chunk, heard),
                   "voice_similarity": r["voice_similarity"], "wav": wav}
            if best is None:
                first_cer = res["cer"]  # kept so we can report how much retrying helped
            if best is None or res["cer"] < best["cer"]:
                best = res
            if best["cer"] <= MAX_CER:
                break
        best.update(attempts=attempt + 1, first_cer=first_cer)
        pieces += [best.pop("wav"), np.zeros(int(0.25 * sr), dtype=np.float32)]  # 0.25 s pause
        report.append(best)
    audio = np.concatenate(pieces) if pieces else np.zeros(1, dtype=np.float32)
    return sr, audio, report


def overall(report):
    """Combine chunk results: CER weighted by chunk length, mean voice similarity."""
    n = sum(len(_clean(r["text"])) for r in report) or 1
    return {"cer": sum(r["cer"] * len(_clean(r["text"])) for r in report) / n,
            "first_cer": sum(r["first_cer"] * len(_clean(r["text"])) for r in report) / n,
            "voice_similarity": float(np.mean([r["voice_similarity"] for r in report])),
            "attempts": sum(r["attempts"] for r in report), "chunks": len(report),
            "heard": " ".join(r["heard"] for r in report)}
