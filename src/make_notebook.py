"""
Builds analysis.ipynb: a notebook that only LOADS cached results from results/
and shows every table and chart. (It does not re-run Whisper.)
Run, then execute:
  python src/make_notebook.py
  jupyter nbconvert --to notebook --execute --inplace analysis.ipynb
"""
import os

import nbformat as nbf

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
nb = nbf.v4.new_notebook()
C = []


def md(s):
    C.append(nbf.v4.new_markdown_cell(s))


def code(s):
    C.append(nbf.v4.new_code_cell(s))


md("""# Accent Gap: analysis notebook
Loads the cached results in `results/` (produced by `src/analyze.py`) and shows every table and chart
used in the README. Nothing here re-runs Whisper. WER = word error rate in %, lower is better.
CIs are 95% bootstrap intervals that resample **speakers**.""")
code("""import pandas as pd
from IPython.display import Image, display
pd.set_option("display.max_colwidth", 200, "display.width", 250, "display.max_rows", 200)
R = "results/"
def show(name, **kw):
    df = pd.read_csv(R + name, keep_default_na=False, **kw)
    display(df)
    return df""")

md("## 1. The data: EdAcc clips and speakers per accent / first language")
code("""ov = pd.read_csv(R + "dataset_overview.csv")
ov[ov.value.str.contains("Indian|Hindi|Urdu|Sinhal|US English|British", regex=True)]""")
md("Evaluation set actually used (last 40% of each speaker's conversation, 1-30 s, max 100 clips per speaker):")
code('show("eval_set_summary.csv");')

md("## 2. E1: the accent gap (whisper-small, plain)")
code('show("wer_by_group.csv"); display(Image(R + "wer_by_group.png"))')
md("How many times higher is each group's WER than native US/UK speakers' (with 95% CI)?")
code('show("accent_gap.csv");')
md("Per-speaker WER (whisper-small, plain). Groups are small, so look at the spread within each group:")
code("""spk = pd.read_csv(R + "wer_by_speaker.csv")
spk[spk.system == "small_plain"].sort_values(["group", "wer_pct"])""")

md("## 3. E2: does a bigger model close the gap?")
code("""wt = pd.read_csv(R + "wer_by_group_all_systems.csv")
e2 = wt[(wt.level == "group") & wt.system.str.endswith("_plain")]
display(e2.pivot(index="name", columns="system", values="wer_pct"))
show("model_speed.csv")
display(Image(R + "model_size.png"))""")

md("""## 4. E3: personal mode
For each speaker: word list from the first 60% of their conversation (text only) → Whisper prompt → scored on the last 40%.
`gain_points` = plain WER − prompted WER (positive = personal mode helped). Control = someone else's list.""")
code('show("personal_mode_by_group.csv"); display(Image(R + "personal_mode.png"))')
md("Does personal mode help South Asian speakers *more* than native speakers? (difference of gains, with CI)")
code('show("personal_mode_gain_comparison.csv");')
md("Per speaker (whisper-small):")
code("""ps = pd.read_csv(R + "personal_mode_by_speaker.csv")
ps[ps.model == "small"].pivot_table(index=["group", "speaker"], columns="prompt", values="gain_points")""")
md("The personal word lists themselves:")
code('pd.read_csv(R + "personal_word_lists.csv")[["speaker", "subgroup", "n_words", "words"]]')

md("Personal mode on all three model sizes (all 40 speakers):")
code("""e3 = pd.read_csv(R + "personal_mode_by_group.csv")
e3[e3.group.str.startswith("All") & e3.prompt.isin(["personal", "control"])]""")
md("Personal-mode design chosen on DEV speakers only (never on eval clips):")
code('show("dev_tuning.csv"); print(open(R + "chosen_design.json").read())')

md("## 4b. E4 (stretch): LoRA fine-tune on DEV speakers only (speaker-disjoint)")
code("""import os
if os.path.exists(R + "lora_vs_plain.csv"):
    show("lora_vs_plain.csv")
else:
    print("E4 not run")""")

md("""## 4c. Demo check: spoken translation in the speaker's voice (`src/voice_check.py`)
CER = character mismatch between the translation and what Whisper-medium heard in the generated audio.
Voice similarity = cosine similarity of speaker embeddings (same encoder the voice model uses, so it may flatter it).""")
code("""import os
if os.path.exists(R + "voice_check_summary.csv"):
    show("voice_check_summary.csv")
    vc = pd.read_csv(R + "voice_check.csv", keep_default_na=False)
    display(vc[["group", "clip_id", "language", "translation", "heard", "cer_first_try_pct", "cer_final_pct",
                "generations", "sim_generated_vs_other_clip", "sim_real_same_speaker", "sim_real_different_speaker"]])
else:
    print("voice check not run")""")

md("## 5. Error analysis (whisper-small)")
md("Words missed most often (substituted or deleted), per group:")
code('show("missed_words_by_group.csv");')
md("Most common substitutions (what was said → what Whisper wrote):")
code('show("top_substitutions_by_group.csv");')
md("Miss rate by word type. 'Rare word' = low frequency in everyday English (proxy for names/places):")
code("""mw = pd.read_csv(R + "miss_rate_by_word_type.csv")
mw.pivot_table(index=["group", "word_type"], columns="system", values="miss_rate_pct")""")
md("WER vs clip length:")
code('show("wer_by_length.csv"); display(Image(R + "wer_by_length.png"))')
md("Inserted words (hallucination check): does prompting make Whisper invent words?")
code("""ins = pd.read_csv(R + "insertions_by_system.csv")
ins.pivot_table(index="group", columns="system", values="insertion_rate_pct")""")
md("Worst clips per group, side by side (reference vs standard vs personal mode):")
code("""w = pd.read_csv(R + "worst_examples.csv", keep_default_na=False)
w[["group", "speaker", "reference", "whisper_small", "personal_mode", "wer_plain", "wer_personal", "why_we_think_it_failed"]]""")

nb["cells"] = C
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
nbf.write(nb, os.path.join(ROOT, "analysis.ipynb"))
print("wrote analysis.ipynb")
