"""
Step 5: turn cached transcripts into every number, table and chart.

Reads   data_cache/clips.csv + results/transcripts_<model>_<mode>.csv
Writes  results/*.csv and results/*.png  (all numbers in the write-up come from here)

Sections:
  E1  accent gap with whisper-small (plain)
  E2  model size: base vs small vs medium
  E3  personal mode vs plain vs control word list
  Error analysis: missed words, rare words / numbers, clip length, worst examples
"""
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from wordfreq import zipf_frequency

from common import CACHE, RESULTS, load_clip, normalize
from metrics import (bootstrap_diff_of_gains, bootstrap_paired_diff, bootstrap_ratio,
                     bootstrap_wer, load_eval_clips, pooled_wer, read_hyps, score_system,
                     with_loop_guard)
from run_transcribe import output_path

GROUPS = ["South Asian English", "Native US/UK English", "Nigerian English",
          "Jamaican English", "Spanish-accented English"]
SA, NATIVE = GROUPS[0], GROUPS[1]
ALL = "All groups (40 speakers)"


def in_group(df, g):
    return df if g == ALL else df[df.group == g]
MODELS = ["base", "small", "medium"]

# Chart style (validated reference palette; see dataviz notes in LEARN.md)
BLUE, ORANGE, AQUA, GRAY = "#2a78d6", "#eb6834", "#1baf7a", "#a3a29c"
TEXT, TEXT2, SURFACE = "#0b0b0b", "#52514e", "#fcfcfb"
plt.rcParams.update({"figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
                     "axes.edgecolor": "#d6d5d0", "axes.labelcolor": TEXT2,
                     "xtick.color": TEXT2, "ytick.color": TEXT2, "text.color": TEXT,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "font.size": 10, "axes.titleweight": "bold"})


def save(df, name):
    df.to_csv(os.path.join(RESULTS, name), index=False)
    print(f"wrote results/{name}")


def pct(x):
    return round(100 * x, 1)


# ------------------------------------------------------------------ tables
def wer_table(df, system):
    """Pooled WER + speaker-bootstrap 95% CI for each group and subgroup."""
    rows = []
    for level in ["group", "subgroup"]:
        for name, d in df.groupby(level):
            lo, hi = bootstrap_wer(d)
            per_spk = d.groupby("speaker").apply(lambda x: x.errors.sum() / x.N.sum(), include_groups=False)
            rows.append({"system": system, "level": level, "group": d.group.iloc[0], "name": name,
                         "speakers": d.speaker.nunique(), "clips": len(d), "ref_words": int(d.N.sum()),
                         "wer_pct": pct(pooled_wer(d)), "ci_low_pct": pct(lo), "ci_high_pct": pct(hi),
                         "mean_speaker_wer_pct": pct(per_spk.mean()),
                         "sub_pct": pct(d.S.sum() / d.N.sum()), "del_pct": pct(d.D.sum() / d.N.sum()),
                         "ins_pct": pct(d.I.sum() / d.N.sum())})
    t = pd.DataFrame(rows)
    t = t[~((t.level == "subgroup") & (t.name == t.group))]  # drop duplicate rows
    return t


def speaker_table(systems):
    rows = []
    for sysname, df in systems.items():
        for spk, d in df.groupby("speaker"):
            rows.append({"system": sysname, "speaker": spk, "group": d.group.iloc[0],
                         "subgroup": d.subgroup.iloc[0], "l1": d.l1.iloc[0], "gender": d.gender.iloc[0],
                         "clips": len(d), "ref_words": int(d.N.sum()), "wer_pct": pct(pooled_wer(d))})
    return pd.DataFrame(rows)


def gap_table(df, system):
    """How many times worse is each group than native US/UK speakers?"""
    rows = []
    nat = df[df.group == NATIVE]
    for g in GROUPS[:1] + GROUPS[2:]:
        d = df[df.group == g]
        lo, hi = bootstrap_ratio(d, nat)
        rows.append({"system": system, "group": g, "group_wer_pct": pct(pooled_wer(d)),
                     "native_wer_pct": pct(pooled_wer(nat)),
                     "ratio_vs_native": round(pooled_wer(d) / pooled_wer(nat), 2),
                     "ratio_ci_low": round(lo, 2), "ratio_ci_high": round(hi, 2)})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ charts
def chart_wer_by_group(t):
    t = t[t.system == "small_plain"].copy()
    order = []
    for g in GROUPS:  # each group, followed by its subgroups (indented)
        order.append((g, t[(t.level == "group") & (t.name == g)].iloc[0], False))
        for _, r in t[(t.level == "subgroup") & (t.group == g)].iterrows():
            order.append((r["name"], r, True))
    fig, ax = plt.subplots(figsize=(8, 0.42 * len(order) + 1.2))
    y = np.arange(len(order))[::-1]
    for yi, (name, r, is_sub) in zip(y, order):
        color = BLUE if r.group == SA else GRAY
        ax.barh(yi, r.wer_pct, height=0.5 if is_sub else 0.7, color=color, alpha=0.55 if is_sub else 1)
        ax.plot([r.ci_low_pct, r.ci_high_pct], [yi, yi], color=TEXT, lw=1.2)
        ax.text(r.ci_high_pct + 0.6, yi, f"{r.wer_pct:.1f}%  ({r.speakers} spk)", va="center",
                fontsize=8.5, color=TEXT2)
    ax.set_yticks(y)
    ax.set_yticklabels([("    " + n) if s else n for n, _, s in order])
    for lbl, (_, _, s) in zip(ax.get_yticklabels(), order):
        lbl.set_fontweight("normal" if s else "bold")
    ax.set_xlabel("Word error rate, % (lower is better). Line = 95% CI, resampling speakers")
    ax.set_title("Whisper-small error rate by accent group (EdAcc held-out clips)", loc="right", fontsize=10.5)
    ax.set_xlim(0, max(t.ci_high_pct) * 1.25)
    ax.grid(axis="x", color="#ecebe7", lw=0.8)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, "wer_by_group.png"), dpi=160)
    plt.close(fig)


def grouped_bars(t, systems, labels, colors, fname, title):
    t = t[t.level == "group"]
    fig, ax = plt.subplots(figsize=(9, 4.4))
    x = np.arange(len(GROUPS))
    w = 0.8 / len(systems)
    for i, (s, lab, c) in enumerate(zip(systems, labels, colors)):
        d = t[t.system == s].set_index("name").reindex(GROUPS)
        if d.wer_pct.isna().all():
            continue
        xs = x - 0.4 + w * (i + 0.5)
        ax.bar(xs, d.wer_pct, width=w * 0.92, color=c, label=lab)
        ax.errorbar(xs, d.wer_pct, yerr=[d.wer_pct - d.ci_low_pct, d.ci_high_pct - d.wer_pct],
                    fmt="none", ecolor=TEXT, elinewidth=1, capsize=0)
        for xi, v in zip(xs, d.wer_pct):
            ax.text(xi, 0.5, f"{v:.1f}", ha="center", va="bottom", fontsize=7.5, color="white",
                    fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([g.replace(" English", "\nEnglish").replace("-accented", "-\naccented") for g in GROUPS],
                       fontsize=9)
    ax.set_ylabel("Word error rate, %")
    ax.set_title(title, loc="left")
    ax.legend(frameon=False, ncol=len(systems), loc="upper left", bbox_to_anchor=(0, 1.0))
    ax.set_ylim(0, t[t.system.isin(systems)].ci_high_pct.max() * 1.18)
    ax.grid(axis="y", color="#ecebe7", lw=0.8)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, fname), dpi=160)
    plt.close(fig)


def chart_length(lt):
    fig, ax = plt.subplots(figsize=(7.5, 4))
    colors = [BLUE, GRAY, ORANGE, AQUA, "#6250d6"]
    for g, c in zip(GROUPS, colors):
        d = lt[lt.group == g]
        ax.plot(d.length_bin.astype(str), d.wer_pct, marker="o", ms=6, lw=2, color=c, label=g)
    ax.set_xlabel("Clip length (seconds)")
    ax.set_ylabel("Word error rate, %")
    ax.set_title("Whisper-small: WER vs clip length", loc="left")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(axis="y", color="#ecebe7", lw=0.8)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, "wer_by_length.png"), dpi=160)
    plt.close(fig)


# ------------------------------------------------------------------ error analysis
def word_category(w):
    if any(ch.isdigit() for ch in w):
        return "number"
    # Rare word = low frequency in everyday English (wordfreq Zipf < 3.5).
    # This catches most names and places (capitol, katniss, hyderabad) but also
    # some rare ordinary words; it is a proxy, not a perfect name detector.
    return "rare word (names, places, ...)" if zipf_frequency(w, "en") < 3.5 else "common word"


def explode_words(df, personal_lists=None):
    rows = []
    for r in df.itertuples():
        plist = personal_lists.get(r.speaker, set()) if personal_lists else set()
        for w, tag in r.word_tags:
            rows.append({"group": r.group, "speaker": r.speaker, "word": w,
                         "missed": tag != "hit", "tag": tag, "in_personal_list": w in plist})
    out = pd.DataFrame(rows)
    out["category"] = out.word.map({w: word_category(w) for w in out.word.unique()})
    return out


def missed_words(words):
    rows = []
    for g in GROUPS:
        d = words[words.group == g]
        agg = d.groupby("word").agg(occurrences=("missed", "size"), missed=("missed", "sum"))
        agg = agg[agg.occurrences >= 3]
        agg["miss_rate_pct"] = (100 * agg.missed / agg.occurrences).round(1)
        top_sub = (d[d.tag.str.startswith("sub:")].groupby("word").tag
                   .agg(lambda s: s.str[4:].value_counts().index[0]))
        agg["most_common_wrong_output"] = top_sub
        agg = agg.sort_values(["missed", "miss_rate_pct"], ascending=False).head(15).reset_index()
        agg.insert(0, "group", g)
        rows.append(agg)
    return pd.concat(rows)


def top_substitutions(words):
    rows = []
    for g in GROUPS:
        d = words[(words.group == g) & words.tag.str.startswith("sub:")]
        c = (d.word + " -> " + d.tag.str[4:]).value_counts().head(15)
        rows.append(pd.DataFrame({"group": g, "reference -> whisper": c.index, "count": c.values}))
    return pd.concat(rows)


def main():
    clips = load_eval_clips()
    print(f"eval clips with scorable text: {len(clips)}")
    systems = {}
    # Final personal-mode design, chosen on DEV speakers only (src/dev_tuning.py)
    design_path = os.path.join(RESULTS, "chosen_design.json")
    final = json.load(open(design_path)) if os.path.exists(design_path) else None
    loop_rows = []
    for m in MODELS:
        plain_h = read_hyps(output_path(m, "plain"))
        df = score_system(clips, hyps=plain_h, name=f"{m}_plain") if plain_h is not None else None
        if df is None:
            continue
        systems[f"{m}_plain"] = df
        for mode in ["personal", "control"]:
            # v1 = our original design (40-word comma list, no loop guard)
            h = read_hyps(output_path(m, mode))
            d1 = score_system(clips, hyps=h, name=f"{m}_{mode}_v1") if h is not None else None
            if d1 is not None:
                systems[f"{m}_{mode}_v1"] = d1
            # final = design chosen on DEV speakers
            if final:
                h = read_hyps(output_path(m, mode, final["style"], final["nwords"]))
                if h is None:
                    continue
                n_loops = 0
                if final["loop_guard"]:
                    h, n_loops = with_loop_guard(h, plain_h)
                d2 = score_system(clips, hyps=h, name=f"{m}_{mode}")
                if d2 is not None:
                    systems[f"{m}_{mode}"] = d2
                    loop_rows.append({"system": f"{m}_{mode}", "clips": len(d2),
                                      "outputs_replaced_by_loop_guard": n_loops})
    if loop_rows:
        save(pd.DataFrame(loop_rows), "loop_guard_counts.csv")
    # E4 (stretch): whisper-small + LoRA trained on DEV speakers only
    h = read_hyps(output_path("small-lora", "plain"))
    d4 = score_system(clips, hyps=h, name="small-lora_plain") if h is not None else None
    if d4 is not None:
        systems["small-lora_plain"] = d4
    print("systems scored:", list(systems))

    # ---- Totals of the evaluation and dev sets (quoted in the README)
    allc = pd.read_csv(os.path.join(CACHE, "clips.csv"))
    tot = []
    for name, role, d in [("eval (40 speakers in 5 groups)", "eval", allc[allc.group != "DEV"]),
                          ("dev (design choices only)", "dev", allc[allc.group == "DEV"])]:
        r = d[d.role == role]
        tot.append({"set": name, "speakers": r.speaker.nunique(), "clips": len(r),
                    "minutes": round(r.duration.sum() / 60, 1),
                    "history_clips_text_only": int((d.role == "history").sum())})
    save(pd.DataFrame(tot), "set_totals.csv")

    # ---- E1 + all systems: WER per group / subgroup / speaker
    wt = pd.concat([wer_table(df, s) for s, df in systems.items()])
    save(wt, "wer_by_group_all_systems.csv")
    save(wt[wt.system == "small_plain"], "wer_by_group.csv")
    save(speaker_table(systems), "wer_by_speaker.csv")
    save(pd.concat([gap_table(df, s) for s, df in systems.items() if s.endswith("plain")]), "accent_gap.csv")
    chart_wer_by_group(wt)

    # ---- E4: LoRA vs plain whisper-small, same clips (paired, resampling speakers)
    if "small-lora_plain" in systems:
        a, b = systems["small_plain"], systems["small-lora_plain"]
        rows = []
        for g in GROUPS + [ALL]:
            da, db = in_group(a, g), in_group(b, g)
            lo, hi = bootstrap_paired_diff(da, db)
            rows.append({"group": g, "speakers": da.speaker.nunique(), "small_wer_pct": pct(pooled_wer(da)),
                         "small_lora_wer_pct": pct(pooled_wer(db)), "gain_points": pct(pooled_wer(da) - pooled_wer(db)),
                         "gain_ci_low": pct(lo), "gain_ci_high": pct(hi),
                         "small_del_pct": pct(da.D.sum() / da.N.sum()), "lora_del_pct": pct(db.D.sum() / db.N.sum())})
        save(pd.DataFrame(rows), "lora_vs_plain.csv")

    # ---- E2: model size
    e2 = [f"{m}_plain" for m in MODELS if f"{m}_plain" in systems]
    grouped_bars(wt, e2, [f"whisper-{s.split('_')[0]}" for s in e2], ["#9ec5f0", BLUE, "#174a87"][:len(e2)],
                 "model_size.png", "Does a bigger Whisper close the accent gap?")
    timing = []
    for s in e2:
        m = s.split("_")[0]
        tr = pd.read_csv(os.path.join(RESULTS, f"transcripts_{m}_plain.csv"))
        timing.append({"model": f"whisper-{m}", "mean_seconds_per_clip": round(tr.seconds.mean(), 3)})
    save(pd.DataFrame(timing), "model_speed.csv")

    # ---- E3: personal mode
    lists = pd.read_csv(os.path.join(RESULTS, "personal_word_lists.csv"), keep_default_na=False)
    plists = {s: set(normalize(w) for w in ws.split(", ") if w) for s, ws in zip(lists.speaker, lists.words)}
    e3_rows, spk_rows, cmp_rows, own_rows = [], [], [], []
    for m in MODELS:
        a = systems.get(f"{m}_plain")
        if a is None:
            continue
        for mode in ["personal", "control", "personal_v1", "control_v1"]:
            b = systems.get(f"{m}_{mode}")
            if b is None:
                continue
            for g in GROUPS + [ALL]:
                da, db = in_group(a, g), in_group(b, g)
                lo, hi = bootstrap_paired_diff(da, db)
                wa, wb = pooled_wer(da), pooled_wer(db)
                ta = da.groupby("speaker").apply(lambda x: x.errors.sum() / x.N.sum(), include_groups=False)
                tb = db.groupby("speaker").apply(lambda x: x.errors.sum() / x.N.sum(), include_groups=False)
                e3_rows.append({"model": m, "prompt": mode, "group": g, "speakers": da.speaker.nunique(),
                                "clips": len(da), "plain_wer_pct": pct(wa), "prompted_wer_pct": pct(wb),
                                "gain_points": pct(wa - wb), "gain_ci_low": pct(lo), "gain_ci_high": pct(hi),
                                "relative_gain_pct": round(100 * (wa - wb) / wa, 1),
                                "speakers_improved": int((tb < ta).sum()),
                                "speakers_worse": int((tb > ta).sum())})
            for spk, d in a.groupby("speaker"):
                db = b[b.speaker == spk]
                spk_rows.append({"model": m, "prompt": mode, "speaker": spk, "group": d.group.iloc[0],
                                 "subgroup": d.subgroup.iloc[0], "clips": len(d),
                                 "plain_wer_pct": pct(pooled_wer(d)), "prompted_wer_pct": pct(pooled_wer(db)),
                                 "gain_points": pct(pooled_wer(d) - pooled_wer(db))})
            # Is the gain bigger for South Asian than native speakers?
            lo, hi = bootstrap_diff_of_gains(a[a.group == SA], b[b.group == SA],
                                             a[a.group == NATIVE], b[b.group == NATIVE])
            gsa = pooled_wer(a[a.group == SA]) - pooled_wer(b[b.group == SA])
            gna = pooled_wer(a[a.group == NATIVE]) - pooled_wer(b[b.group == NATIVE])
            cmp_rows.append({"model": m, "prompt": mode, "gain_south_asian_points": pct(gsa),
                             "gain_native_points": pct(gna), "difference_points": pct(gsa - gna),
                             "difference_ci_low": pct(lo), "difference_ci_high": pct(hi)})
        # Own list vs someone else's list (same prompt format): is it the
        # PERSONAL words that help, or just having any prompt?
        p, c = systems.get(f"{m}_personal"), systems.get(f"{m}_control")
        if p is not None and c is not None:
            for g in GROUPS + [ALL]:
                dc, dp = in_group(c, g), in_group(p, g)
                lo, hi = bootstrap_paired_diff(dc, dp)
                own_rows.append({"model": m, "group": g, "speakers": dp.speaker.nunique(),
                                 "control_wer_pct": pct(pooled_wer(dc)), "personal_wer_pct": pct(pooled_wer(dp)),
                                 "personal_better_by_points": pct(pooled_wer(dc) - pooled_wer(dp)),
                                 "ci_low": pct(lo), "ci_high": pct(hi)})
    if own_rows:
        save(pd.DataFrame(own_rows), "personal_vs_control.csv")
    if e3_rows:
        save(pd.DataFrame(e3_rows), "personal_mode_by_group.csv")
        save(pd.DataFrame(spk_rows), "personal_mode_by_speaker.csv")
        save(pd.DataFrame(cmp_rows), "personal_mode_gain_comparison.csv")
    e3sys = [s for s in ["small_plain", "small_personal", "small_control"] if s in systems]
    grouped_bars(wt, e3sys, ["Standard Whisper", "Personal mode (own word list)",
                             "Control (someone else's list)"][:len(e3sys)], [BLUE, ORANGE, AQUA][:len(e3sys)],
                 "personal_mode.png", "Personal mode vs standard Whisper-small (held-out clips)")

    # ---- Error analysis (whisper-small)
    a = systems["small_plain"]
    words_a = explode_words(a, plists)
    save(missed_words(words_a), "missed_words_by_group.csv")
    save(top_substitutions(words_a), "top_substitutions_by_group.csv")

    cat_rows = []
    for s in ["small_plain", "small_personal", "small_control", "small_personal_v1"]:
        if s not in systems:
            continue
        w = explode_words(systems[s], plists)
        for g in GROUPS:
            d = w[w.group == g]
            for cat, dc in d.groupby("category"):
                cat_rows.append({"system": s, "group": g, "word_type": cat, "occurrences": len(dc),
                                 "miss_rate_pct": pct(dc.missed.mean())})
            dp = d[d.in_personal_list]
            cat_rows.append({"system": s, "group": g, "word_type": "word in speaker's personal list",
                             "occurrences": len(dp), "miss_rate_pct": pct(dp.missed.mean()) if len(dp) else None})
            dn = d[~d.in_personal_list]
            cat_rows.append({"system": s, "group": g, "word_type": "word NOT in personal list",
                             "occurrences": len(dn), "miss_rate_pct": pct(dn.missed.mean())})
    save(pd.DataFrame(cat_rows), "miss_rate_by_word_type.csv")

    # WER vs clip length
    bins = [1, 3, 6, 10, 30]
    a["length_bin"] = pd.cut(a.duration, bins=bins, labels=["1-3 s", "3-6 s", "6-10 s", "10-30 s"],
                             include_lowest=True)
    lt = (a.groupby(["group", "length_bin"], observed=True)
          .apply(lambda d: pd.Series({"clips": len(d), "wer_pct": pct(pooled_wer(d))}), include_groups=False)
          .reset_index())
    save(lt, "wer_by_length.csv")
    chart_length(lt)

    # Hallucination-like outputs: many inserted words (often caused by prompts)
    hall = []
    for s, df in systems.items():
        for g in GROUPS:
            d = df[df.group == g]
            hall.append({"system": s, "group": g, "clips": len(d),
                         "clips_with_5plus_insertions": int((d.I >= 5).sum()),
                         "empty_outputs": int((d.hyp.str.replace(r"[^A-Za-z]", "", regex=True) == "").sum()),
                         "prompt_echo_outputs": int(d.hyp.str.lower().str.contains("words i often use").sum()),
                         "insertion_rate_pct": pct(d.I.sum() / d.N.sum())})
    save(pd.DataFrame(hall), "insertions_by_system.csv")

    # Loudness of each clip (dB relative to full scale; -20 is normal speech,
    # -45 is very quiet). Helps explain clips where Whisper outputs nothing.
    a["audio_level_db"] = [round(20 * np.log10(np.sqrt(np.mean(load_clip(c) ** 2)) + 1e-9), 1)
                           for c in a.clip_id]

    # Worst examples, side by side
    side = a[["group", "speaker", "clip_id", "duration", "audio_level_db", "ref", "hyp", "N", "wer"]].rename(
        columns={"ref": "reference", "hyp": "whisper_small", "N": "ref_words", "wer": "wer_plain"})
    if "small_personal" in systems:
        p = systems["small_personal"][["clip_id", "hyp", "wer"]].rename(
            columns={"hyp": "personal_mode", "wer": "wer_personal"})
        side = side.merge(p, on="clip_id")
    worst = (side[side.ref_words >= 5].sort_values("wer_plain", ascending=False)
             .groupby("group").head(15).sort_values(["group", "wer_plain"], ascending=[True, False]))
    for c in ["wer_plain", "wer_personal"]:
        if c in worst:
            worst[c] = (100 * worst[c]).round(1)
    worst["why_we_think_it_failed"] = ""
    worst["listen_check_needed"] = ""
    # Hand-written explanations (our best guess, to be checked by listening)
    notes_path = os.path.join(os.path.dirname(RESULTS), "notes", "failure_notes.csv")
    if os.path.exists(notes_path):
        notes = pd.read_csv(notes_path, keep_default_na=False).set_index("clip_id")
        for c in ["why_we_think_it_failed", "listen_check_needed"]:
            worst[c] = worst.clip_id.map(notes[c]).fillna("")
        worst["picked_for_writeup"] = worst.clip_id.isin(notes.index)
    save(worst, "worst_examples.csv")
    save(side.assign(wer_plain=(100 * side.wer_plain).round(1),
                     wer_personal=(100 * side.wer_personal).round(1) if "wer_personal" in side else None),
         "all_clips_side_by_side.csv")


if __name__ == "__main__":
    main()
