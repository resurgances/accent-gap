"""
Shared helpers used by every script: paths, text cleaning, audio loading,
and running Whisper. Kept in one place so every experiment does things the
exact same way (important for a fair comparison).
"""
import io
import os
import re
import zlib

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")
CACHE = os.path.join(ROOT, "data_cache")
AUDIO_DIR = os.path.join(CACHE, "audio")  # 16 kHz clips saved as .npy (git-ignored)
os.makedirs(RESULTS, exist_ok=True)
os.makedirs(AUDIO_DIR, exist_ok=True)

SAMPLE_RATE = 16000  # Whisper expects 16,000 samples per second, mono


# ---------------------------------------------------------------- text cleaning
def clean_reference(text):
    """Remove EdAcc annotation tags so the reference is only spoken words.

    EdAcc transcripts contain tags like <laugh>, <overlap>, <dtmf>, [noise],
    and a special marker IGNORE_TIME_SEGMENT_IN_SCORING. None of these are words
    a person said, so Whisper should not be punished for not typing them.
    """
    if not isinstance(text, str):
        return ""
    if "IGNORE_TIME_SEGMENT_IN_SCORING" in text:
        return ""
    text = re.sub(r"<[^>]*>", " ", text)    # <laugh>, <overlap>, ...
    text = re.sub(r"\[[^\]]*\]", " ", text)  # [noise], ...
    return re.sub(r"\s+", " ", text).strip()


_normalizer = None


def normalize(text):
    """Whisper's official English normalizer: lowercases, removes punctuation,
    standardises spellings and numbers ("ten" vs "10", "colour" vs "color"),
    and removes filler words like "um". We apply it to BOTH the reference and
    the model output before computing WER, so we only count real word errors.
    """
    global _normalizer
    if _normalizer is None:
        from transformers import WhisperTokenizer
        _normalizer = WhisperTokenizer.from_pretrained("openai/whisper-small")
    return _normalizer.normalize(text or "")


# ---------------------------------------------------------------- audio
def decode_audio_bytes(wav_bytes):
    """Turn raw WAV file bytes into a 16 kHz mono float array."""
    import librosa
    import soundfile as sf
    audio, sr = sf.read(io.BytesIO(wav_bytes), dtype="float32", always_2d=True)
    audio = audio.mean(axis=1)  # stereo -> mono
    if sr != SAMPLE_RATE:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=SAMPLE_RATE)
    return audio.astype(np.float32)


def audio_path(clip_id):
    return os.path.join(AUDIO_DIR, clip_id + ".npy")


def load_clip(clip_id):
    return np.load(audio_path(clip_id))


# ---------------------------------------------------------------- whisper
def get_device():
    import torch
    return "mps" if torch.backends.mps.is_available() else "cpu"


def load_whisper(size="small"):
    """Load a pretrained Whisper model (no training by us)."""
    import torch
    from transformers import WhisperForConditionalGeneration, WhisperProcessor
    lora = size.endswith("-lora")  # E4: whisper-small + our LoRA adapter (lora_out/)
    name = f"openai/whisper-{size.replace('-lora', '')}"
    processor = WhisperProcessor.from_pretrained(name)
    model = WhisperForConditionalGeneration.from_pretrained(name, dtype=torch.float32)
    if lora:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, os.path.join(ROOT, "lora_out")).merge_and_unload()
    model.to(get_device()).eval()
    return processor, model


def transcribe(processor, model, audios, prompt=None, language="english"):
    """Transcribe a list of 16 kHz audio arrays (a batch) with Whisper.

    prompt: optional text (our "personal word list"). Whisper treats it as if
    it were the text spoken just before this clip, which nudges it toward
    those words and spellings. This is PROMPTING, not training: the model's
    weights never change.
    """
    import torch
    device = model.device
    feats = processor.feature_extractor(audios, sampling_rate=SAMPLE_RATE,
                                        return_tensors="pt").input_features.to(device)
    kwargs = dict(language=language, task="transcribe", max_new_tokens=200)
    if prompt:
        kwargs["prompt_ids"] = torch.tensor(processor.get_prompt_ids(prompt)).to(device)
    with torch.no_grad():
        out = model.generate(feats, **kwargs)
    texts = processor.batch_decode(out, skip_special_tokens=True)
    texts = [t.strip() for t in texts]
    if prompt:
        # Safety net: if the prompt text is echoed at the start, strip it.
        p = prompt.strip()
        texts = [t[len(p):].strip() if t.startswith(p) else t for t in texts]
    return texts


def looks_like_loop(text):
    """True if Whisper's output looks like a repetition loop ("no, no, no, ...").

    Same test the original Whisper code uses: compress the text with zlib. Normal
    sentences barely shrink; text that repeats itself shrinks a lot. A
    compression ratio above 2.4 is Whisper's own threshold for "this decode failed".
    """
    b = (text or "").encode("utf-8")
    return len(b) > 0 and len(b) / len(zlib.compress(b)) > 2.4
