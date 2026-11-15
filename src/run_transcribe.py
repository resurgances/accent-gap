"""
Step 4: run Whisper on every EVAL (or DEV) clip and cache the text it produces.

Usage:
  python run_transcribe.py --model small --mode plain     # E1 baseline ("A")
  python run_transcribe.py --model small --mode personal  # E3 personal mode ("B")
  python run_transcribe.py --model small --mode control   # E3 control: someone else's list
  python run_transcribe.py --model base  --mode plain     # E2 model size
Options:
  --set dev            run on the DEV speakers (used only to choose design details)
  --style list|sentence --nwords N   how the word list is turned into a prompt

Output: results/transcripts_<model>_<mode>[_<style><N>][_dev].csv
        (clip_id, hyp, prompt, seconds)
The script is RESUMABLE: it appends rows as it goes, and on restart skips
clips that are already done.
"""
import argparse
import os
import time

import pandas as pd

from common import CACHE, RESULTS, load_clip, load_whisper, transcribe


def make_prompt(words_str, style, nwords):
    """Turn a stored word list ("cool, nice, food, ...") into the prompt text."""
    words = [w for w in words_str.split(", ") if w][:nwords]
    if not words:
        return None
    if style == "list":       # v1: plain comma-separated list
        return ", ".join(words)
    if style == "sentence":   # a short natural sentence containing the words
        return "Words I often use: " + ", ".join(words) + "."
    raise ValueError(style)


def output_path(model, mode, style="list", nwords=40, dataset="eval"):
    name = f"transcripts_{model}_{mode}"
    if mode != "plain" and (style, nwords) != ("list", 40):
        name += f"_{style}{nwords}"
    if dataset == "dev":
        name += "_dev"
    return os.path.join(RESULTS, name + ".csv")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="small")
    ap.add_argument("--mode", default="plain", choices=["plain", "personal", "control"])
    ap.add_argument("--set", dest="dataset", default="eval", choices=["eval", "dev"])
    ap.add_argument("--style", default="list", choices=["list", "sentence"])
    ap.add_argument("--nwords", type=int, default=40)
    ap.add_argument("--limit", type=int, default=None, help="only N clips (for quick tests)")
    args = ap.parse_args()

    clips = pd.read_csv(os.path.join(CACHE, "clips.csv"))
    clips = clips[clips.role == args.dataset].sort_values(["speaker", "part", "clip_num"])
    if args.limit:
        clips = clips.head(args.limit)

    prompts = {}
    if args.mode != "plain":
        lists = pd.read_csv(os.path.join(RESULTS, "personal_word_lists.csv"), keep_default_na=False)
        col = "words" if args.mode == "personal" else "control_words"
        prompts = {s: make_prompt(w, args.style, args.nwords) for s, w in zip(lists.speaker, lists[col])}

    out_csv = output_path(args.model, args.mode, args.style, args.nwords, args.dataset)
    done = set(pd.read_csv(out_csv).clip_id) if os.path.exists(out_csv) else set()
    todo = clips[~clips.clip_id.isin(done)]
    print(f"{os.path.basename(out_csv)}: {len(done)} done, {len(todo)} to go", flush=True)
    if len(todo) == 0:
        return

    processor, model = load_whisper(args.model)
    buf, t_start = [], time.time()
    for i, r in enumerate(todo.itertuples()):
        prompt = prompts.get(r.speaker)
        t = time.time()
        hyp = transcribe(processor, model, [load_clip(r.clip_id)], prompt=prompt)[0]
        buf.append({"clip_id": r.clip_id, "hyp": hyp, "prompt": prompt or "",
                    "seconds": round(time.time() - t, 3)})
        if len(buf) >= 25 or i == len(todo) - 1:
            pd.DataFrame(buf).to_csv(out_csv, mode="a", header=not os.path.exists(out_csv), index=False)
            buf = []
            rate = (time.time() - t_start) / (i + 1)
            print(f"  {i + 1}/{len(todo)}  {rate:.2f} s/clip  ETA {rate * (len(todo) - i - 1) / 60:.1f} min",
                  flush=True)


if __name__ == "__main__":
    main()
