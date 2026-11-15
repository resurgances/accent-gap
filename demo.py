"""
Accent Gap demo (Gradio). Run:  .venv/bin/python demo.py   then open the printed URL.

Tab 1 "Speak": record yourself. When you stop recording, it automatically:
  1. detects which language you spoke (Whisper)
  2. writes down what you said (Whisper). With "learn my words" on, it builds a
     word list from what you said EARLIER in this session and uses it as a hint
     (our "personal mode", same recipe as in the README, no typing needed)
  3. translates it into the language you picked (NLLB-200 translation model)
  4. SPEAKS the translation in your own voice (Chatterbox Multilingual voice cloning),
     then checks it: Whisper listens to the generated speech and we compare what it
     heard with the translation; parts that don't match are regenerated (src/voice.py)
Tab 2 "Real EdAcc examples": real held-out clips from the public EdAcc dataset,
  so the demo works without a microphone (needs data_cache/ from the pipeline).

PROTOTYPE: a research demo, not a product. See README for what was measured.
"""
import difflib
import html
import os
import re
import sys

import gradio as gr
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
from common import (CACHE, RESULTS, SAMPLE_RATE, get_device, load_clip,  # noqa: E402
                    load_whisper, looks_like_loop, normalize, transcribe)
from personal_words import personal_list  # noqa: E402
from run_transcribe import make_prompt  # noqa: E402
import voice  # noqa: E402

# Personal-mode design from the README (chosen on DEV speakers): sentence prompt, 40 words,
# loop guard. It helped whisper-small a little but HURT whisper-medium in our tests,
# so "learn my words" is only applied with the Fast (small) model.
STYLE, NWORDS = "sentence", 40
MODELS = {"Fast (whisper-small)": "small", "Most accurate (whisper-medium, slower)": "medium"}
_whisper = {}
_nllb = {}

# Whisper language code -> NLLB-200 language code
WHISPER_TO_NLLB = {
    "en": "eng_Latn", "fr": "fra_Latn", "es": "spa_Latn", "de": "deu_Latn", "it": "ita_Latn",
    "pt": "por_Latn", "nl": "nld_Latn", "pl": "pol_Latn", "ru": "rus_Cyrl", "tr": "tur_Latn",
    "ar": "arb_Arab", "hi": "hin_Deva", "ur": "urd_Arab", "sd": "snd_Arab", "pa": "pan_Guru",
    "gu": "guj_Gujr", "bn": "ben_Beng", "ta": "tam_Taml", "te": "tel_Telu", "mr": "mar_Deva",
    "ne": "npi_Deva", "si": "sin_Sinh", "zh": "zho_Hans", "ja": "jpn_Jpan", "ko": "kor_Hang",
    "vi": "vie_Latn", "id": "ind_Latn", "sw": "swh_Latn", "yo": "yor_Latn", "tl": "tgl_Latn",
}
LANG_NAMES = {
    "en": "English", "fr": "French", "es": "Spanish", "de": "German", "it": "Italian",
    "pt": "Portuguese", "nl": "Dutch", "pl": "Polish", "ru": "Russian", "tr": "Turkish",
    "ar": "Arabic", "hi": "Hindi", "ur": "Urdu", "sd": "Sindhi", "pa": "Punjabi",
    "gu": "Gujarati", "bn": "Bengali", "ta": "Tamil", "te": "Telugu", "mr": "Marathi",
    "ne": "Nepali", "si": "Sinhala", "zh": "Chinese (Mandarin)", "ja": "Japanese", "ko": "Korean",
    "vi": "Vietnamese", "id": "Indonesian", "sw": "Swahili", "yo": "Yoruba", "tl": "Tagalog",
}
NO_TRANSLATION = "Don't translate"
TARGETS = [NO_TRANSLATION] + sorted(LANG_NAMES.values())
NAME_TO_CODE = {v: k for k, v in LANG_NAMES.items()}

NOTE = ("**Prototype.** Built for the Veritas AI Hackathon 2026 (AI + Languages). Speech-to-text = "
        "OpenAI Whisper. Translation = Meta NLLB-200. *Learn my words* = our personal mode: it remembers words "
        "you used earlier in this session and gives them to Whisper as a hint (no voice training). In our tests "
        "this gave a small improvement with whisper-small. Translation quality was not measured by us. "
        "*Spoken translation* = Resemble AI Chatterbox Multilingual: it copies the voice from the recording; "
        "the audio is **AI-generated** and invisibly watermarked by the model. Only copy a voice with that "
        "person's permission.")


# ------------------------------------------------------------------ models (loaded on first use)
def whisper(size):
    if size not in _whisper:
        print(f"loading whisper-{size} ...")
        _whisper[size] = load_whisper(size)
    return _whisper[size]


def nllb():
    if not _nllb:
        print("loading NLLB-200 translation model (first time downloads ~2.5 GB) ...")
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        name = "facebook/nllb-200-distilled-600M"
        _nllb["tok"] = AutoTokenizer.from_pretrained(name)
        _nllb["model"] = AutoModelForSeq2SeqLM.from_pretrained(name).to(get_device()).eval()
    return _nllb["tok"], _nllb["model"]


def detect_language(size, x):
    """Ask Whisper which language is being spoken. Returns a code like 'en'."""
    processor, model = whisper(size)
    feats = processor.feature_extractor([x], sampling_rate=SAMPLE_RATE,
                                        return_tensors="pt").input_features.to(model.device)
    token = processor.tokenizer.decode(model.detect_language(feats)[0])  # e.g. "<|en|>"
    return token.strip("<|>")


def translate(text, src_code, target_name):
    """Translate text sentence by sentence (NLLB sometimes drops a sentence when given several)."""
    import torch
    if target_name == NO_TRANSLATION or not text.strip():
        return ""
    tgt_code = NAME_TO_CODE[target_name]
    if tgt_code == src_code:
        return text
    if src_code not in WHISPER_TO_NLLB:
        return f"(Sorry, translating from the detected language '{src_code}' is not set up in this demo.)"
    tok, model = nllb()
    tok.src_lang = WHISPER_TO_NLLB[src_code]
    sentences = [s for s in re.split(r"(?<=[.!?।؟])\s+", text.strip()) if s]
    batch = tok(sentences, return_tensors="pt", padding=True).to(model.device)
    with torch.no_grad():
        out = model.generate(**batch, forced_bos_token_id=tok.convert_tokens_to_ids(WHISPER_TO_NLLB[tgt_code]),
                             max_new_tokens=200)
    return " ".join(tok.batch_decode(out, skip_special_tokens=True))


# ------------------------------------------------------------------ helpers
def to_16k_mono(audio):
    """Gradio gives (sample_rate, int16 array). Whisper wants 16 kHz mono floats."""
    import librosa
    sr, x = audio
    x = x.astype(np.float32)
    if x.ndim == 2:
        x = x.mean(axis=1)
    if np.abs(x).max() > 1.0:
        x = x / 32768.0
    if sr != SAMPLE_RATE:
        x = librosa.resample(x, orig_sr=sr, target_sr=SAMPLE_RATE)
    return x[: 30 * SAMPLE_RATE]  # Whisper looks at max 30 s at a time


def personal_mode(size, x, words_str, language="english"):
    """Standard output, plus personal-mode output with the loop guard
    (if the personal output looks like a repetition loop, fall back to standard)."""
    processor, model = whisper(size)
    standard = transcribe(processor, model, [x], language=language)[0]
    prompt = make_prompt(words_str, STYLE, NWORDS)
    if not prompt:
        return standard, standard, ""
    personal = transcribe(processor, model, [x], prompt=prompt, language=language)[0]
    if looks_like_loop(personal):
        return standard, standard, "(loop guard: the hinted output was a repetition loop, so we kept Standard)"
    return standard, personal, ""


def highlight(base_words, other_words, color):
    """Highlight words in `other_words` that differ from `base_words`; show missing words crossed out."""
    sm = difflib.SequenceMatcher(a=base_words, b=other_words, autojunk=False)
    out = []
    for op, i1, i2, j1, j2 in sm.get_opcodes():
        if op == "equal":
            out += [html.escape(w) for w in other_words[j1:j2]]
        else:
            if op in ("replace", "insert"):
                out += [f'<mark style="background:{color};color:#0b0b0b;padding:0 2px;border-radius:3px">'
                        f'{html.escape(w)}</mark>' for w in other_words[j1:j2]]
            if op in ("replace", "delete"):
                out += [f'<s style="opacity:0.6">{html.escape(w)}</s>' for w in base_words[i1:i2]]
    return " ".join(out)


def word_errors(ref, hyp):
    import jiwer
    r, h = normalize(ref), normalize(hyp)
    if not r.strip():
        return None
    return jiwer.wer(r, h) if h.strip() else 1.0


def compare_html(standard, personal, reference="", left="Standard Whisper", right="With learned words"):
    """Side-by-side panel. With a reference, highlight each side's mistakes vs the
    reference. Without one, highlight where the right side differs from the left."""
    cell = 'style="vertical-align:top;padding:10px;border:1px solid #d6d5d0;width:50%"'
    if reference.strip():
        ref_w = normalize(reference).split()
        s_html = highlight(ref_w, normalize(standard).split(), "#f6c9b8")
        p_html = highlight(ref_w, normalize(personal).split(), "#f6c9b8")
        ws, wp = word_errors(reference, standard), word_errors(reference, personal)
        if ws is not None:
            left += f" — WER {100 * ws:.0f}%"
        if wp is not None:
            right += f" — WER {100 * wp:.0f}%"
        legend = ("Highlighted = word that does not match what was said; crossed out = word that was "
                  "said but missed. (Text is normalized: lowercase, no punctuation.)")
        ref_row = f'<p><b>What was actually said:</b> {html.escape(reference)}</p>'
    else:
        s_html = html.escape(standard)
        p_html = highlight(standard.split(), personal.split(), "#bfe8d6")
        legend = "Highlighted = where the right side wrote something different from the left."
        ref_row = ""
    return (f"{ref_row}<table style='width:100%;border-collapse:collapse'><tr>"
            f"<th {cell}>{left}</th><th {cell}>{right}</th></tr>"
            f"<tr><td {cell}>{s_html}</td><td {cell}>{p_html}</td></tr></table>"
            f"<p style='font-size:0.85em;opacity:0.75'>{legend}</p>")


# ------------------------------------------------------------------ spoken translation
def listen(x, lang):
    """Accuracy check: Whisper-medium transcribes the generated speech in the target language.
    (Medium, not small: on a Hindi test, small misheard much more, which made good audio look bad.)"""
    import torch
    processor, model = whisper("medium")
    heard = transcribe(processor, model, [x], language=lang)[0]
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()  # free Apple-GPU memory between checks
    return heard


def voice_anchor():
    """Reference numbers for the voice-match score, measured by src/voice_check.py (if it was run)."""
    path = os.path.join(RESULTS, "voice_check_summary.csv")
    if not os.path.exists(path):
        return ""
    d = pd.read_csv(path).set_index("measure").value
    same, diff = float(d["real_same_speaker_similarity_mean"]), float(d["real_different_speaker_similarity_mean"])
    return (f" For comparison, on EdAcc: two *real* recordings of the same speaker scored "
            f"{same:.2f} on average, two different speakers {diff:.2f}.")


def speak_translation(ref_audio, text, target):
    """Returns (audio for gr.Audio, markdown report)."""
    if not text or target == NO_TRANSLATION or text.startswith("(Sorry"):
        return None, ""
    lang = NAME_TO_CODE[target]
    if not voice.available():
        return None, "Spoken translation needs the voice env: see README (`.venv-tts`)."
    if lang not in voice.SUPPORTED:
        names = {**LANG_NAMES, "da": "Danish", "el": "Greek", "fi": "Finnish", "he": "Hebrew",
                 "ms": "Malay", "no": "Norwegian", "sv": "Swedish"}
        return None, (f"The voice model can't speak {target} (the text translation above still works). "
                      "It speaks: " + ", ".join(sorted(names[c] for c in voice.SUPPORTED)) + ".")
    sr, audio, report = voice.speak(ref_audio, text, lang, listen)
    o = voice.overall(report)
    warn = ("\n\n**Warning: some of it still didn't match after 3 tries, so don't trust this audio.**"
            if any(r["cer"] > voice.MAX_CER for r in report) else "")
    md = (f"**Accuracy check:** Whisper listened to the spoken translation and heard:  \n*{html.escape(o['heard'])}*  \n"
          f"Character mismatch with the translation: **{100 * o['cer']:.0f}%** "
          f"(parts above {100 * voice.MAX_CER:.0f}% are regenerated, up to {voice.MAX_TRIES} tries; "
          f"{o['attempts']} generation(s) for {o['chunks']} part(s)).  \n"
          f"**Voice match:** {o['voice_similarity']:.2f} (1.0 = same voice fingerprint as the recording).{voice_anchor()}"
          f"{warn}  \n<sub>This checks that the audio says the translated text. It does not check that the "
          f"translation itself is right.</sub>")
    return (sr, audio), md


def voice_sample(clips):
    """The voice to copy: newest recording first, plus earlier ones, up to ~10 s."""
    out, total = [], 0
    for c in reversed(clips):
        out.append(c)
        total += len(c)
        if total >= 10 * SAMPLE_RATE:
            break
    return np.concatenate(out)


# ------------------------------------------------------------------ tab 1: speak
def run_speak(audio, target, model_label, learn, history, voice_clips):
    """Runs automatically when you stop recording (or upload a file)."""
    history = history or []
    voice_clips = voice_clips or []
    if audio is None:
        return ("Record something first.", "", "", ", ".join(personal_list(history)) or "(none yet)", "",
                history, voice_clips)
    size = MODELS[model_label]
    x = to_16k_mono(audio)
    lang = detect_language(size, x)
    lang_name = LANG_NAMES.get(lang, lang)
    learned = personal_list(history)  # words from what you said EARLIER (never this clip)

    note = ""
    if learn and lang == "en" and size == "small" and learned:
        standard, said, note = personal_mode(size, x, ", ".join(learned), language="en")
        compare = compare_html(standard, said)
    else:
        processor, model = whisper(size)
        said = transcribe(processor, model, [x], language=lang)[0]
        if learn and size != "small":
            note = "(Learned words are not used with whisper-medium: in our tests they made it worse.)"
        elif learn and lang != "en":
            note = "(Learned words are only used for English, which is what we tested.)"
        elif learn and not learned:
            note = "(Nothing learned yet: words you say at least twice will be remembered.)"
        compare = ""

    history = history + [said]  # remember this for next time
    voice_clips = voice_clips + [x]  # your voice, to copy when speaking the translation
    detected = f"Detected language: **{lang_name}**" + (f"  \n{note}" if note else "")
    return (said, translate(said, lang, target), detected, ", ".join(personal_list(history)) or "(none yet)",
            compare, history, voice_clips)


def speak_mine(translation, target, speak_it, voice_clips):
    if not speak_it or not voice_clips:
        return None, ""
    return speak_translation(voice_sample(voice_clips), translation, target)


def reset_learning():
    return [], [], "(none yet)", ""


# ------------------------------------------------------------------ tab 2: EdAcc examples
def load_examples():
    side_path = os.path.join(RESULTS, "all_clips_side_by_side.csv")
    lists_path = os.path.join(RESULTS, "personal_word_lists.csv")
    if not (os.path.exists(side_path) and os.path.exists(lists_path)):
        return pd.DataFrame(), {}
    side = pd.read_csv(side_path, keep_default_na=False)
    side = side[side.clip_id.apply(lambda c: os.path.exists(os.path.join(CACHE, "audio", c + ".npy")))]
    # 1) the clips discussed in the README (notes/failure_notes.csv) come first
    notes_path = os.path.join(os.path.dirname(RESULTS), "notes", "failure_notes.csv")
    noted = pd.read_csv(notes_path).clip_id.tolist() if os.path.exists(notes_path) else []
    picks = [side.set_index("clip_id").loc[[c for c in noted if c in set(side.clip_id)]].reset_index()]
    # 2) plus automatic picks per group (6-30 words long)
    side = side[(side.ref_words >= 6) & (side.ref_words <= 30)]
    for g, d in side.groupby("group"):
        d = d.assign(gain=d.wer_plain - d.wer_personal)
        picks += [d.sort_values("gain", ascending=False).head(3),   # learned words helped most
                  d.sort_values("gain").head(1),                    # learned words hurt most
                  d[d.wer_plain > 0].sample(min(2, (d.wer_plain > 0).sum()), random_state=0)]  # typical
    ex = pd.concat(picks).drop_duplicates("clip_id")
    ex["label"] = [("★ " if r.clip_id in noted else "") +
                   f"{r.group} | {r.clip_id} | standard WER {r.wer_plain:.0f}%, with learned words {r.wer_personal:.0f}%"
                   for r in ex.itertuples()]
    lists = pd.read_csv(lists_path, keep_default_na=False)
    return ex.set_index("label"), dict(zip(lists.speaker, lists.words))


EXAMPLES, WORD_LISTS = load_examples()
CLIPS = pd.read_csv(os.path.join(CACHE, "clips.csv")) if os.path.exists(os.path.join(CACHE, "clips.csv")) else None


def speaker_voice(clip_id, speaker):
    """Voice to copy for an EdAcc clip: the clip itself, plus other cached clips of the
    same speaker if it is shorter than voice.MIN_REF_SECONDS. (Audio only, for the voice.)"""
    x = [load_clip(clip_id)]
    if CLIPS is not None:
        for c in CLIPS[(CLIPS.speaker == speaker) & (CLIPS.clip_id != clip_id)].clip_id:
            if sum(map(len, x)) >= voice.MIN_REF_SECONDS * SAMPLE_RATE:
                break
            if os.path.exists(os.path.join(CACHE, "audio", c + ".npy")):
                x.append(load_clip(c))
    return np.concatenate(x)[: 10 * SAMPLE_RATE]


def show_example(label):
    if not label:
        return None, "", "", "", "", None, ""
    r = EXAMPLES.loc[label]
    return (SAMPLE_RATE, load_clip(r.clip_id)), r.reference, WORD_LISTS.get(r.speaker, ""), "", "", None, ""


def run_example(label, target):
    if not label:
        return "Pick an example first.", ""
    r = EXAMPLES.loc[label]
    standard, personal, note = personal_mode("small", load_clip(r.clip_id), WORD_LISTS.get(r.speaker, ""))
    panel = compare_html(standard, personal, r.reference) + (f"<p><i>{note}</i></p>" if note else "")
    return panel, translate(personal, "en", target)


def speak_example(label, translation, target, speak_it):
    if not label or not speak_it:
        return None, ""
    r = EXAMPLES.loc[label]
    return speak_translation(speaker_voice(r.clip_id, r.speaker), translation, target)


# ------------------------------------------------------------------ UI
with gr.Blocks(title="Accent Gap") as demo:
    gr.Markdown("# Accent Gap: speak in your own accent, get your words (in any language)")
    gr.Markdown(NOTE)
    history = gr.State([])
    voice_clips = gr.State([])  # your recordings this session (the voice to copy)

    with gr.Tab("Speak"):
        gr.Markdown("Press record, talk, press stop. It transcribes automatically. "
                    "The more you talk, the more of your words it learns.")
        with gr.Row():
            with gr.Column():
                audio_in = gr.Audio(sources=["microphone", "upload"], type="numpy",
                                    label="Speak (max 30 s per recording)")
                target = gr.Dropdown(TARGETS, value="French", label="Translate to")
                model_choice = gr.Radio(list(MODELS), value=list(MODELS)[0], label="Speech model")
                learn = gr.Checkbox(value=True, label="Learn my words as I talk (personal mode)")
                speak_it = gr.Checkbox(value=True, label="Speak the translation in my voice")
                with gr.Row():
                    again = gr.Button("Transcribe again", variant="secondary")
                    reset = gr.Button("Forget learned words", variant="secondary")
            with gr.Column():
                said_out = gr.Textbox(label="What you said", lines=3)
                trans_out = gr.Textbox(label="Translation", lines=3)
                voice_out = gr.Audio(label="Translation spoken in your voice (AI-generated)", autoplay=True,
                                     interactive=False)
                voice_info = gr.Markdown()
                lang_out = gr.Markdown()
                learned_out = gr.Textbox(label="Words it has learned from you so far", value="(none yet)",
                                         interactive=False)
                with gr.Accordion("Compare: without vs with learned words", open=False):
                    cmp_out = gr.HTML()
        inputs = [audio_in, target, model_choice, learn, history, voice_clips]
        outputs = [said_out, trans_out, lang_out, learned_out, cmp_out, history, voice_clips]
        # first the text (fast), then the spoken translation (slower)
        for event in (audio_in.stop_recording, audio_in.upload, again.click):
            event(run_speak, inputs, outputs).then(
                speak_mine, [trans_out, target, speak_it, voice_clips], [voice_out, voice_info])
        reset.click(reset_learning, None, [history, voice_clips, learned_out, cmp_out])

    with gr.Tab("Real EdAcc examples (no mic needed)"):
        if len(EXAMPLES) == 0:
            gr.Markdown("No cached examples found. Run `src/prepare_data.py`, `run_all.sh` and `src/analyze.py` first.")
        else:
            gr.Markdown("Held-out clips from public EdAcc speakers. The learned words come automatically from "
                        "**earlier** in the same speaker's conversation (never from this clip). "
                        "★ = clips discussed in the README.")
            pick = gr.Dropdown(list(EXAMPLES.index), label="Choose a clip")
            ex_target = gr.Dropdown(TARGETS, value="French", label="Translate to")
            ex_speak = gr.Checkbox(value=True, label="Speak the translation in the speaker's voice")
            ex_audio = gr.Audio(label="Clip", type="numpy", interactive=False)
            ex_ref = gr.Textbox(label="Reference transcript (written by a human)")
            ex_words = gr.Textbox(label="Words learned from earlier in this conversation (automatic)")
            ex_go = gr.Button("Run: transcribe, translate, speak", variant="primary")
            ex_cmp = gr.HTML()
            ex_trans = gr.Textbox(label="Translation (of the 'with learned words' text)")
            ex_voice = gr.Audio(label="Translation spoken in this speaker's voice (AI-generated)", autoplay=True,
                                interactive=False)
            ex_voice_info = gr.Markdown()
            pick.change(show_example, pick, [ex_audio, ex_ref, ex_words, ex_cmp, ex_trans, ex_voice, ex_voice_info])
            ex_go.click(run_example, [pick, ex_target], [ex_cmp, ex_trans]).then(
                speak_example, [pick, ex_trans, ex_target, ex_speak], [ex_voice, ex_voice_info])

if __name__ == "__main__":
    whisper("small")  # load the fast model up front so the first recording is quick
    if voice.available():
        voice._worker()  # start the voice model now too (takes ~30 s)
        whisper("medium")  # used to check the spoken translation
    demo.launch()
