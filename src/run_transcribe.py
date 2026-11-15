"""
Step 4: run Whisper on every EVAL clip and cache the text it produces.

Usage:
  python run_transcribe.py --model small --mode plain     # E1 baseline ("A")
  python run_transcribe.py --model small --mode personal  # E3 personal mode ("B")
  python run_transcribe.py --model small --mode control   # E3 control: someone else's list
  python run_transcribe.py --model base  --mode plain     # E2 model size

Output: results/transcripts_<model>_<mode>.csv  (clip_id, hyp, prompt, seconds)
The script is RESUMABLE: it appends one row per clip, and on restart skips
clips that are already done.
"""
import argparse
import os
import time

import pandas as pd

from common import CACHE, RESULTS, load_clip, load_whisper, transcribe


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="small")
    ap.add_argument("--mode", default="plain", choices=["plain", "personal", "control"])
    ap.add_argument("--limit", type=int, default=None, help="only N clips (for quick tests)")
    args = ap.parse_args()

    clips = pd.read_csv(os.path.join(CACHE, "clips.csv"))
    clips = clips[clips.role == "eval"].sort_values(["speaker", "part", "clip_num"])
    if args.limit:
        clips = clips.head(args.limit)

    prompts = {}
    if args.mode != "plain":
        lists = pd.read_csv(os.path.join(RESULTS, "personal_word_lists.csv"), keep_default_na=False)
        col = "words" if args.mode == "personal" else "control_words"
        prompts = dict(zip(lists.speaker, lists[col]))

    out_csv = os.path.join(RESULTS, f"transcripts_{args.model}_{args.mode}.csv")
    done = set(pd.read_csv(out_csv).clip_id) if os.path.exists(out_csv) else set()
    todo = clips[~clips.clip_id.isin(done)]
    print(f"{args.model}/{args.mode}: {len(done)} done, {len(todo)} to go", flush=True)

    processor, model = load_whisper(args.model)
    buf, t_start = [], time.time()
    for i, r in enumerate(todo.itertuples()):
        prompt = prompts.get(r.speaker) or None
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
