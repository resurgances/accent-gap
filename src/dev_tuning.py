"""
Choose the personal-mode design on the DEV speakers only.

Why: our first design (v1 = 40-word comma list) made Whisper fall into
repetition loops. Trying fixes on the evaluation clips and keeping the best
one would be cheating (we'd be tuning on the test). So we compare variants on
DEV speakers (Irish, Scottish, Kenyan, South African, Italian, Vietnamese -
none of them are in any reported group), pick the winner here, and run it
ONCE on the evaluation set.

Variants: prompt style (list / sentence) x list length (40 / 15) x loop guard (off / on).
Output: results/dev_tuning.csv
"""
import json
import os

import pandas as pd

from common import RESULTS
from metrics import load_eval_clips, pooled_wer, read_hyps, score_system, with_loop_guard


def main():
    clips = load_eval_clips(role="dev")
    plain_h = read_hyps(os.path.join(RESULTS, "transcripts_small_plain_dev.csv"))
    plain = score_system(clips, hyps=plain_h, name="plain")
    if plain is None:
        print("dev plain run not finished")
        return
    plain_spk = plain.groupby("speaker").apply(lambda d: d.errors.sum() / d.N.sum(), include_groups=False)
    rows = [{"variant": "plain (no prompt)", "style": "", "nwords": 0, "loop_guard": False,
             "speakers": plain.speaker.nunique(), "clips": len(plain), "wer_pct": round(100 * pooled_wer(plain), 1),
             "gain_points": 0.0, "insertion_pct": round(100 * plain.I.sum() / plain.N.sum(), 1),
             "loops_replaced": 0, "speakers_improved": 0, "speakers_worse": 0}]
    for style in ["list", "sentence"]:
        for n in [40, 15]:
            suffix = "" if (style, n) == ("list", 40) else f"_{style}{n}"
            h = read_hyps(os.path.join(RESULTS, f"transcripts_small_personal{suffix}_dev.csv"))
            if h is None:
                continue
            for guard in [False, True]:
                hh, n_loops = with_loop_guard(h, plain_h) if guard else (h, 0)
                df = score_system(clips, hyps=hh, name=f"{style}{n}")
                if df is None:
                    continue
                spk = df.groupby("speaker").apply(lambda d: d.errors.sum() / d.N.sum(), include_groups=False)
                rows.append({"variant": f"{style} of {n} words" + (" + loop guard" if guard else ""),
                             "style": style, "nwords": n, "loop_guard": guard,
                             "speakers": df.speaker.nunique(), "clips": len(df),
                             "wer_pct": round(100 * pooled_wer(df), 1),
                             "gain_points": round(100 * (pooled_wer(plain) - pooled_wer(df)), 1),
                             "insertion_pct": round(100 * df.I.sum() / df.N.sum(), 1),
                             "loops_replaced": n_loops,
                             "speakers_improved": int((spk < plain_spk).sum()),
                             "speakers_worse": int((spk > plain_spk).sum())})
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(RESULTS, "dev_tuning.csv"), index=False)
    print(out.to_string(index=False))

    # Rule fixed in advance: pick the prompted variant with the lowest DEV WER.
    prompted = out[out.nwords > 0]
    if len(prompted) == 8:  # all 4 prompt variants x guard on/off finished
        best = prompted.sort_values("wer_pct").iloc[0]
        choice = {"style": best["style"], "nwords": int(best.nwords), "loop_guard": bool(best.loop_guard),
                  "dev_wer_pct": float(best.wer_pct), "dev_plain_wer_pct": float(out.wer_pct.iloc[0])}
        json.dump(choice, open(os.path.join(RESULTS, "chosen_design.json"), "w"), indent=2)
        print("chosen design:", choice)


if __name__ == "__main__":
    main()
