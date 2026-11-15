"""
Scoring helpers: Word Error Rate (WER) and bootstrap confidence intervals.

WER = (substitutions + deletions + insertions) / number of words in the reference
  substitution: Whisper wrote a different word   ("capitol" -> "capital")
  deletion:     Whisper skipped a word
  insertion:    Whisper added a word that was not said
WER 0.10 = about 1 error per 10 words. Lower is better.
"""
import os

import jiwer
import numpy as np
import pandas as pd

from common import CACHE, RESULTS, normalize

N_BOOT = 2000  # number of bootstrap resamples


def score_clip(ref_norm, hyp_norm):
    """Align reference and hypothesis words. Returns error counts and, for every
    reference word, what happened to it: 'hit', 'sub:<hyp word>' or 'del'."""
    ref_words = ref_norm.split()
    if not hyp_norm.strip():  # Whisper produced nothing: every word is deleted
        return dict(S=0, D=len(ref_words), I=0, N=len(ref_words)), [(w, "del") for w in ref_words]
    out = jiwer.process_words(ref_norm, hyp_norm)
    tags = []
    r, h = out.references[0], out.hypotheses[0]
    for ch in out.alignments[0]:
        if ch.type == "equal":
            tags += [(r[i], "hit") for i in range(ch.ref_start_idx, ch.ref_end_idx)]
        elif ch.type == "substitute":
            for k, i in enumerate(range(ch.ref_start_idx, ch.ref_end_idx)):
                tags.append((r[i], "sub:" + h[ch.hyp_start_idx + k]))
        elif ch.type == "delete":
            tags += [(r[i], "del") for i in range(ch.ref_start_idx, ch.ref_end_idx)]
    counts = dict(S=out.substitutions, D=out.deletions, I=out.insertions, N=len(r))
    return counts, tags


def load_eval_clips():
    clips = pd.read_csv(os.path.join(CACHE, "clips.csv"))
    clips = clips[clips.role == "eval"].copy()
    clips["ref_norm"] = clips.ref.apply(normalize)
    # Clips whose reference is ONLY filler words (e.g. "UM", "MM HMM") become
    # empty after normalization and cannot be scored -> excluded from ALL systems.
    return clips[clips.ref_norm.str.strip() != ""]


def score_system(clips, model, mode):
    """Score one system (e.g. small/plain). Returns per-clip table, or None if
    that system has not been run yet / is incomplete."""
    path = os.path.join(RESULTS, f"transcripts_{model}_{mode}.csv")
    if not os.path.exists(path):
        return None
    hyp = pd.read_csv(path, keep_default_na=False).drop_duplicates("clip_id")
    df = clips.merge(hyp[["clip_id", "hyp"]], on="clip_id", how="inner")
    if len(df) < len(clips):
        print(f"  {model}/{mode}: only {len(df)}/{len(clips)} clips done - skipping")
        return None
    df["hyp_norm"] = df.hyp.apply(normalize)
    res = [score_clip(r, h) for r, h in zip(df.ref_norm, df.hyp_norm)]
    for k in "SDIN":
        df[k] = [c[k] for c, _ in res]
    df["errors"] = df.S + df.D + df.I
    df["wer"] = df.errors / df.N
    df["word_tags"] = [t for _, t in res]
    df["system"] = f"{model}_{mode}"
    return df


def pooled_wer(df):
    """Pooled WER: total errors / total reference words (long clips count more)."""
    return df.errors.sum() / df.N.sum()


def speaker_totals(df):
    return df.groupby("speaker")[["errors", "N"]].sum()


def bootstrap_wer(df, seed=0):
    """95% confidence interval for a group's pooled WER, resampling SPEAKERS.

    Why speakers and not clips? Clips from the same person are not independent
    (same voice, same microphone). Resampling whole speakers asks: "if we had
    recorded a different set of people like these, how much could the number
    change?" With few speakers the interval is wide - that is honest.
    """
    t = speaker_totals(df)
    rng = np.random.default_rng(seed)
    e, n = t.errors.values, t.N.values
    idx = rng.integers(0, len(t), size=(N_BOOT, len(t)))
    boots = e[idx].sum(1) / n[idx].sum(1)
    return np.percentile(boots, 2.5), np.percentile(boots, 97.5)


def bootstrap_ratio(df_a, df_b, seed=0):
    """CI for WER(group a) / WER(group b), resampling speakers in each group."""
    rng = np.random.default_rng(seed)
    ta, tb = speaker_totals(df_a), speaker_totals(df_b)
    ia = rng.integers(0, len(ta), size=(N_BOOT, len(ta)))
    ib = rng.integers(0, len(tb), size=(N_BOOT, len(tb)))
    wa = ta.errors.values[ia].sum(1) / ta.N.values[ia].sum(1)
    wb = tb.errors.values[ib].sum(1) / tb.N.values[ib].sum(1)
    r = wa / wb
    return np.percentile(r, 2.5), np.percentile(r, 97.5)


def paired_speaker_table(df_a, df_b):
    """Per-speaker error totals for two systems run on the SAME clips."""
    a = speaker_totals(df_a).rename(columns={"errors": "err_a"})
    b = speaker_totals(df_b)[["errors"]].rename(columns={"errors": "err_b"})
    return a.join(b)


def bootstrap_paired_diff(df_a, df_b, seed=0):
    """CI for WER(a) - WER(b) on the same clips, resampling speakers (paired)."""
    t = paired_speaker_table(df_a, df_b)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(t), size=(N_BOOT, len(t)))
    n = t.N.values[idx].sum(1)
    d = (t.err_a.values[idx].sum(1) - t.err_b.values[idx].sum(1)) / n
    return np.percentile(d, 2.5), np.percentile(d, 97.5)


def bootstrap_diff_of_gains(a1, b1, a2, b2, seed=0):
    """CI for [gain in group 1] - [gain in group 2], gain = WER(a) - WER(b).
    Used to ask: does personal mode help South Asian speakers MORE than natives?"""
    rng = np.random.default_rng(seed)
    out = []
    t1, t2 = paired_speaker_table(a1, b1), paired_speaker_table(a2, b2)
    i1 = rng.integers(0, len(t1), size=(N_BOOT, len(t1)))
    i2 = rng.integers(0, len(t2), size=(N_BOOT, len(t2)))
    g1 = (t1.err_a.values[i1].sum(1) - t1.err_b.values[i1].sum(1)) / t1.N.values[i1].sum(1)
    g2 = (t2.err_a.values[i2].sum(1) - t2.err_b.values[i2].sum(1)) / t2.N.values[i2].sum(1)
    d = g1 - g2
    return np.percentile(d, 2.5), np.percentile(d, 97.5)
