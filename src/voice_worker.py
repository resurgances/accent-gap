"""
Voice worker: speaks a translation in the SAME voice as a reference recording.

Runs in its own Python env (.venv-tts), because the voice-cloning model
(Chatterbox Multilingual by Resemble AI, MIT license) pins library versions that
clash with the main .venv. demo.py starts this file once and talks to it through
stdin/stdout, one JSON request per line:

  request : {"ref_wav": "voice.wav", "text": "Bonjour", "lang": "fr", "out_wav": "out.wav"}
  reply   : {"ok": true, "out_wav": "out.wav", "voice_similarity": 0.83, "seconds": 4.1}
  or      : {"compare": ["a.wav", "b.wav"]} -> {"ok": true, "voice_similarity": 0.71}

How the cloning works (no training): the model listens to the reference clip,
turns it into a "speaker embedding" (a list of numbers describing the voice),
and generates new speech for the text conditioned on that embedding.

voice_similarity = cosine similarity between the speaker embedding of the
reference and of the generated audio (same voice encoder the model uses).
1.0 = identical voice embedding. It is a rough automatic check, not a human rating.

Every generated file carries Resemble's inaudible "Perth" watermark, which marks
it as AI-generated.
"""
import json
import sys
import time

import librosa
import numpy as np
import soundfile as sf
import torch

SUPPORTED = {"ar", "da", "de", "el", "en", "es", "fi", "fr", "he", "hi", "it", "ja", "ko", "ms", "nl",
             "no", "pl", "pt", "ru", "sv", "sw", "tr", "zh"}


# stdout is reserved for JSON replies. Libraries sometimes print status lines, so we
# point sys.stdout at stderr and keep the real stdout only for our replies.
REPLY = sys.stdout
sys.stdout = sys.stderr


def log(*a):
    print(*a, file=sys.stderr, flush=True)


def reply(obj):
    REPLY.write(json.dumps(obj) + "\n")
    REPLY.flush()


def load_model():
    from chatterbox.mtl_tts import ChatterboxMultilingualTTS
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    try:
        return ChatterboxMultilingualTTS.from_pretrained(device=device)
    except Exception as e:  # some ops are not implemented on Apple GPUs: fall back to CPU
        log(f"loading on {device} failed ({e}); using cpu")
        return ChatterboxMultilingualTTS.from_pretrained(device="cpu")


def limit_length(model, text, lang):
    """The model normally may generate up to 1000 speech tokens (~40 s) for any text, and
    sometimes babbles on to that limit. Speech tokens come at 25 per second; even slow
    speech covers ~8 characters per second (~3 for Chinese/Japanese/Korean, where one
    character is a whole syllable or word), so allow 25 * (chars/rate + 2) tokens at most."""
    rate = 3 if lang in ("zh", "ja", "ko") else 8
    limit = min(1000, int(25 * (len(text) / rate + 2)))
    if not hasattr(model.t3, "_original_inference"):
        model.t3._original_inference = model.t3.inference

    def inference(*a, **k):
        k["max_new_tokens"] = limit
        return model.t3._original_inference(*a, **k)
    model.t3.inference = inference


MAX_GPU_GB = 8  # if the model's Apple-GPU memory grows past this, restart the worker


def free_gpu_memory():
    """Apple-GPU memory otherwise piles up between requests (we saw the Mac swap heavily).
    Returns how much the GPU driver still holds for this process, in GB."""
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
        return torch.mps.driver_allocated_memory() / 1e9
    return 0.0


def voice_embedding(model, wav, sr):
    x = librosa.resample(wav.astype(np.float32), orig_sr=sr, target_sr=16000)
    e = model.ve.embeds_from_wavs([x], sample_rate=16000).mean(axis=0)
    return e / np.linalg.norm(e)


def main():
    log("loading Chatterbox Multilingual (first time downloads ~3 GB) ...")
    model = load_model()
    reply({"ready": True})
    for line in sys.stdin:
        try:
            req = json.loads(line)
            if "compare" in req:  # voice similarity between two existing recordings (used by voice_check.py)
                (a, sa), (b, sb) = [librosa.load(f, sr=None) for f in req["compare"]]
                reply({"ok": True, "voice_similarity": float(voice_embedding(model, a, sa) @ voice_embedding(model, b, sb))})
                continue
            if req["lang"] not in SUPPORTED:
                raise ValueError(f"language '{req['lang']}' is not supported by the voice model")
            t0 = time.time()
            torch.manual_seed(req.get("seed", 0))  # generation is random: fix the seed per attempt
            limit_length(model, req["text"], req["lang"])
            wav = model.generate(req["text"], language_id=req["lang"], audio_prompt_path=req["ref_wav"],
                                 cfg_weight=req.get("cfg_weight", 0.5),
                                 exaggeration=req.get("exaggeration", 0.5))
            wav = wav.squeeze(0).numpy()
            sf.write(req["out_wav"], wav, model.sr)
            ref, ref_sr = librosa.load(req["ref_wav"], sr=None)
            sim = float(voice_embedding(model, ref, ref_sr) @ voice_embedding(model, wav, model.sr))
            gpu_gb = free_gpu_memory()
            reply({"ok": True, "out_wav": req["out_wav"], "voice_similarity": sim,
                   "seconds": time.time() - t0, "gpu_gb": gpu_gb})
        except Exception as e:
            gpu_gb = free_gpu_memory()
            reply({"ok": False, "error": str(e)})
        if gpu_gb > MAX_GPU_GB:  # memory crept up: exit; voice.py starts a fresh worker next time
            log(f"voice worker using {gpu_gb:.1f} GB of GPU memory, restarting")
            break


if __name__ == "__main__":
    main()
