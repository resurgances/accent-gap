"""
Step 2: choose speakers and clips, then download + cache ONLY the audio we need.

Groups (decided per speaker from EdAcc's metadata):
  South Asian English  = accent "Indian English" OR first language (l1) is a
                         South Asian language (Hindi, Urdu, Sinhalese, ...).
                         Sub-groups keep accent / l1 separate.
  Native US English    = accent and l1 both "Mainstream US English"
  Native UK English    = accent and l1 both "Southern British English"
  Nigerian / Jamaican / Spanish-accented English = context groups

The personal-mode split (core of the project, decided here so EVERY experiment
uses the same clips):
  Each EdAcc speaker = one recorded conversation. We sort their clips in
  conversation order. The FIRST 60% are the speaker's "history" (we only ever
  use their reference TEXT to build a personal word list). The LAST 40% are
  the "eval" clips. All experiments (E1 baseline, E2 model size, E3 personal
  mode) are scored ONLY on eval clips, so no word list ever sees test text.

Outputs:
  data_cache/clips.csv         every clip of chosen speakers with role history/eval
  data_cache/audio/<id>.npy    16 kHz audio for eval clips (git-ignored)
  results/eval_set_summary.csv clips / speakers / minutes per group
"""
import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from huggingface_hub import HfFileSystem

from common import CACHE, RESULTS, SAMPLE_RATE, audio_path, clean_reference, decode_audio_bytes

HISTORY_FRACTION = 0.6     # first 60% of a conversation = personal history
MAX_EVAL_PER_SPEAKER = 100  # cap so one talkative speaker can't dominate a group
MIN_CLIPS_PER_SPEAKER = 20  # need enough clips for both history and eval
MIN_SEC, MAX_SEC = 1.0, 30.0

SOUTH_ASIAN_L1 = {"Hindi", "Urdu", "Gujarati", "Sindhi", "Punjabi", "Marathi", "Bengali",
                  "Tamil", "Telugu", "Kannada", "Malayalam", "Nepali", "Sinhalese",
                  "Indian English"}

DUR_LOG = os.path.join(CACHE, "durations.csv")


def assign_group(accent, l1, raw_accent):
    """Return (group, subgroup) for a speaker, or (None, None) if not used."""
    if accent == "Indian English" or l1 in SOUTH_ASIAN_L1:
        sub = "Indian English accent" if accent == "Indian English" else f"{l1} L1"
        return "South Asian English", sub
    if accent == "Mainstream US English" and l1 == "Mainstream US English" \
            and "pakistan" not in str(raw_accent).lower():
        return "Native US/UK English", "US English"
    if accent == "Southern British English" and l1 == "Southern British English":
        return "Native US/UK English", "UK English"
    if accent == "Nigerian English":
        return "Nigerian English", "Nigerian English"
    if accent == "Jamaican English":
        return "Jamaican English", "Jamaican English"
    if accent == "Spanish":
        return "Spanish-accented English", "Spanish-accented English"
    return None, None


def build_clip_table():
    meta = pd.read_csv(os.path.join(CACHE, "metadata.csv"))
    meta["ref"] = meta["text"].apply(clean_reference)
    meta = meta[meta["ref"] != ""]  # drop empty / IGNORE / tag-only clips

    rows = []
    for spk, d in meta.groupby("speaker"):
        first = d.iloc[0]
        group, sub = assign_group(first.accent, first.l1, first.raw_accent)
        if group is None:
            continue
        if len(d) < MIN_CLIPS_PER_SPEAKER:
            print(f"skip {spk} ({group}): only {len(d)} clips")
            continue
        # Conversation order. Some conversations were recorded in parts
        # (clip ids like EDACC-C31_P1-12, EDACC-C31_P2-3) and clip numbers
        # restart in each part, so sort by (part, clip number).
        d = d.copy()
        d["part"] = d.clip_id.str.extract(r"_P(\d+)-")[0].fillna(0).astype(int)
        d = d.sort_values(["part", "clip_num"])
        n_hist = int(round(len(d) * HISTORY_FRACTION))
        d["role"] = ["history"] * n_hist + ["eval_candidate"] * (len(d) - n_hist)
        d["group"], d["subgroup"] = group, sub
        rows.append(d)
    return pd.concat(rows)


def download_row_group(shard, rg, wanted_ids):
    """Download one chunk of a parquet file, save the clips we want as .npy."""
    pf = pq.ParquetFile(shard, filesystem=HfFileSystem())
    t = pf.read_row_group(int(rg), columns=["audio"]).to_pylist()
    out = []
    for r in t:
        cid = r["audio"]["path"].replace(".wav", "")
        if cid not in wanted_ids:
            continue
        audio = decode_audio_bytes(r["audio"]["bytes"])
        dur = len(audio) / SAMPLE_RATE
        if MIN_SEC <= dur <= MAX_SEC:
            np.save(audio_path(cid), audio)
        out.append((cid, dur))
    return out


def main():
    clips = build_clip_table()
    cand = clips[clips.role == "eval_candidate"]

    done = pd.read_csv(DUR_LOG) if os.path.exists(DUR_LOG) else pd.DataFrame(columns=["clip_id", "duration"])
    todo = cand[~cand.clip_id.isin(done.clip_id)]
    jobs = list(todo.groupby(["shard", "row_group"]))
    print(f"eval candidates: {len(cand)}, still to download: {len(todo)} in {len(jobs)} chunks")

    def run(job):
        (shard, rg), d = job
        return download_row_group(shard, rg, set(d.clip_id))

    with ThreadPoolExecutor(max_workers=4) as ex:
        for i, res in enumerate(ex.map(run, jobs)):
            pd.DataFrame(res, columns=["clip_id", "duration"]).to_csv(
                DUR_LOG, mode="a", header=not os.path.exists(DUR_LOG), index=False)
            if i % 10 == 0:
                print(f"  chunk {i + 1}/{len(jobs)}", flush=True)

    durs = pd.read_csv(DUR_LOG).drop_duplicates("clip_id")
    clips = clips.merge(durs, on="clip_id", how="left")

    # Final eval set: eval candidates with 1-30 s audio, then (if a speaker has
    # more than the cap) keep an evenly spaced subset across their eval part.
    ok = (clips.role == "eval_candidate") & clips.duration.between(MIN_SEC, MAX_SEC)
    clips.loc[clips.role == "eval_candidate", "role"] = "eval_dropped"
    for spk, d in clips[ok].groupby("speaker"):
        idx = d.index
        if len(idx) > MAX_EVAL_PER_SPEAKER:
            idx = idx[np.linspace(0, len(idx) - 1, MAX_EVAL_PER_SPEAKER).round().astype(int)]
        clips.loc[idx, "role"] = "eval"

    cols = ["clip_id", "speaker", "group", "subgroup", "accent", "l1", "gender", "split",
            "part", "clip_num", "role", "duration", "ref"]
    clips[cols].to_csv(os.path.join(CACHE, "clips.csv"), index=False)

    ev = clips[clips.role == "eval"]
    summ = ev.groupby(["group", "subgroup"]).agg(
        speakers=("speaker", "nunique"), eval_clips=("clip_id", "count"),
        eval_minutes=("duration", lambda x: round(x.sum() / 60, 1))).reset_index()
    hist = clips[clips.role == "history"].groupby(["group", "subgroup"]).size().rename("history_clips")
    summ = summ.merge(hist, on=["group", "subgroup"])
    summ.to_csv(os.path.join(RESULTS, "eval_set_summary.csv"), index=False)
    print(summ.to_string(index=False))
    print(ev.groupby(["group", "speaker"]).size().to_string())


if __name__ == "__main__":
    main()
