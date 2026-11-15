"""
Accent Gap demo (Gradio). Run:  .venv/bin/python demo.py   then open the printed URL.

Tab 1 "Try it": record or upload audio, optionally type a personal word list,
  and compare Standard Whisper vs Personal mode side by side.
Tab 2 "Real EdAcc examples": play real held-out clips from the EdAcc test
  material (works without a microphone). Needs data_cache/ from prepare_data.py.

PROTOTYPE: this is a research demo, not a product. Personal mode only uses a
word list as a prompt; it does not train on anyone's voice.
"""
import difflib
import html
import os
import sys

import gradio as gr
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
from common import (CACHE, RESULTS, SAMPLE_RATE, load_clip, load_whisper, looks_like_loop,  # noqa: E402
                    normalize, transcribe)
from run_transcribe import make_prompt  # noqa: E402

MODEL_SIZE = "small"
print(f"loading whisper-{MODEL_SIZE} ...")
PROCESSOR, MODEL = load_whisper(MODEL_SIZE)

# Same personal-mode design as in the README (chosen on DEV speakers):
# the word list goes into a short sentence prompt, plus the loop guard.
STYLE, NWORDS = "sentence", 40


def personal_mode(x, words_str):
    """Personal mode = Whisper + word-list prompt, with the loop guard:
    if the output looks like a repetition loop, fall back to standard output."""
    prompt = make_prompt(words_str, STYLE, NWORDS)
    standard = transcribe(PROCESSOR, MODEL, [x])[0]
    if not prompt:
        return standard, standard, "(no word list given, so Personal mode = Standard)"
    personal = transcribe(PROCESSOR, MODEL, [x], prompt=prompt)[0]
    if looks_like_loop(personal):
        return standard, standard, "(loop guard: personal output was a repetition loop, fell back to Standard)"
    return standard, personal, ""


NOTE = ("**Prototype.** Built for the Veritas AI Hackathon 2026 (AI + Languages). "
        "Standard = openai/whisper-small. Personal mode = the same model, given a short list of words "
        "this person often uses as a text prompt (no voice training). Results in our README come from "
        "public EdAcc speakers, not from the people who inspired this project.")


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


def highlight(base_words, other_words, color):
    """Words in `other_words` that differ from `base_words` get highlighted;
    words that are missing are shown crossed out."""
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
                out += [f'<s style="color:#a3a29c">{html.escape(w)}</s>' for w in base_words[i1:i2]]
    return " ".join(out)


def word_errors(ref, hyp):
    import jiwer
    r, h = normalize(ref), normalize(hyp)
    if not r.strip():
        return None
    return jiwer.wer(r, h) if h.strip() else 1.0


def compare_html(standard, personal, reference=""):
    """Side-by-side panel. With a reference, highlight each system's mistakes vs
    the reference. Without one, highlight where personal mode differs."""
    cell = 'style="vertical-align:top;padding:10px;border:1px solid #d6d5d0;width:50%"'
    if reference.strip():
        ref_w = normalize(reference).split()
        s_html = highlight(ref_w, normalize(standard).split(), "#f6c9b8")
        p_html = highlight(ref_w, normalize(personal).split(), "#f6c9b8")
        ws, wp = word_errors(reference, standard), word_errors(reference, personal)
        head_s = f"Standard Whisper — WER {100 * ws:.0f}%" if ws is not None else "Standard Whisper"
        head_p = f"Personal mode — WER {100 * wp:.0f}%" if wp is not None else "Personal mode"
        legend = ("Highlighted = word that does not match what was said; "
                  "<s>crossed out</s> = word that was said but missed. (Text is normalized: lowercase, no punctuation.)")
        ref_row = f'<p><b>What was actually said:</b> {html.escape(reference)}</p>'
    else:
        s_html = html.escape(standard)
        p_html = highlight(standard.split(), personal.split(), "#bfe8d6")
        head_s, head_p = "Standard Whisper", "Personal mode"
        legend = "Highlighted = where Personal mode wrote something different from Standard."
        ref_row = ""
    return (f"{ref_row}<table style='width:100%;border-collapse:collapse'><tr>"
            f"<th {cell}>{head_s}</th><th {cell}>{head_p}</th></tr>"
            f"<tr><td {cell}>{s_html}</td><td {cell}>{p_html}</td></tr></table>"
            f"<p style='font-size:0.85em;opacity:0.75'>{legend}</p>")


# ------------------------------------------------------------------ tab 1
def run_live(audio, word_list, reference):
    if audio is None:
        return "Please record or upload audio first.", "", ""
    x = to_16k_mono(audio)
    words = ", ".join(w.strip() for w in (word_list or "").replace("\n", ",").split(",") if w.strip())
    standard, personal, note = personal_mode(x, words)
    return standard, (personal + " " + note).strip(), compare_html(standard, personal, reference or "") + \
        (f"<p><i>{note}</i></p>" if note else "")


# ------------------------------------------------------------------ tab 2
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
        picks += [d.sort_values("gain", ascending=False).head(3),   # personal mode helped most
                  d.sort_values("gain").head(1),                    # personal mode hurt most
                  d[d.wer_plain > 0].sample(min(2, (d.wer_plain > 0).sum()), random_state=0)]  # typical errors
    ex = pd.concat(picks).drop_duplicates("clip_id")
    ex["label"] = [("★ " if r.clip_id in noted else "") +
                   f"{r.group} | {r.clip_id} | standard WER {r.wer_plain:.0f}%, personal {r.wer_personal:.0f}%"
                   for r in ex.itertuples()]
    lists = pd.read_csv(lists_path, keep_default_na=False)
    return ex.set_index("label"), dict(zip(lists.speaker, lists.words))


EXAMPLES, WORD_LISTS = load_examples()


def show_example(label):
    if not label:
        return None, "", "", ""
    r = EXAMPLES.loc[label]
    audio = (SAMPLE_RATE, load_clip(r.clip_id))
    return audio, r.reference, WORD_LISTS.get(r.speaker, ""), ""


def run_example(label):
    if not label:
        return "Pick an example first."
    r = EXAMPLES.loc[label]
    standard, personal, note = personal_mode(load_clip(r.clip_id), WORD_LISTS.get(r.speaker, ""))
    return compare_html(standard, personal, r.reference) + (f"<p><i>{note}</i></p>" if note else "")


# ------------------------------------------------------------------ UI
with gr.Blocks(title="Accent Gap") as demo:
    gr.Markdown("# Accent Gap: does speech-to-text understand everyone equally?")
    gr.Markdown(NOTE)
    with gr.Tab("Try it (your voice)"):
        with gr.Row():
            with gr.Column():
                audio_in = gr.Audio(sources=["microphone", "upload"], type="numpy", label="Speak or upload (max 30 s)")
                words_in = gr.Textbox(label="Personal word list (optional, comma separated)",
                                      placeholder="e.g. Karachi, Hyderabad, biryani, Ammi, Diwali, dentist")
                ref_in = gr.Textbox(label="What you actually said (optional, enables error highlighting)")
                go = gr.Button("Transcribe", variant="primary")
            with gr.Column():
                std_out = gr.Textbox(label="Standard Whisper")
                per_out = gr.Textbox(label="Personal mode")
                cmp_out = gr.HTML()
        go.click(run_live, [audio_in, words_in, ref_in], [std_out, per_out, cmp_out])

    with gr.Tab("Real EdAcc examples (no mic needed)"):
        if len(EXAMPLES) == 0:
            gr.Markdown("No cached examples found. Run `src/prepare_data.py`, `run_all.sh` and `src/analyze.py` first.")
        else:
            gr.Markdown("Held-out clips from public EdAcc speakers. The personal word list was built from "
                        "**earlier** parts of the same speaker's conversation only (never from this clip).")
            pick = gr.Dropdown(list(EXAMPLES.index), label="Choose a clip")
            ex_audio = gr.Audio(label="Clip", type="numpy", interactive=False)
            ex_ref = gr.Textbox(label="Reference transcript (human-written)")
            ex_words = gr.Textbox(label="This speaker's personal word list")
            ex_go = gr.Button("Run Standard vs Personal mode", variant="primary")
            ex_cmp = gr.HTML()
            pick.change(show_example, pick, [ex_audio, ex_ref, ex_words, ex_cmp])
            ex_go.click(run_example, pick, ex_cmp)

if __name__ == "__main__":
    demo.launch()
