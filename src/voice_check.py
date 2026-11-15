"""
Measures the demo's spoken translation (voice cloning + accuracy check) on EdAcc clips.
Writes results/voice_check.csv (one row per clip x language) and
results/voice_check_summary.csv (averages). Run:  .venv/bin/python src/voice_check.py

For each sampled held-out EdAcc clip and each target language, we run exactly what the
demo does: Whisper transcript (with learned words, from results/) -> NLLB translation ->
Chatterbox speaks it in the speaker's voice -> Whisper-medium listens and we compute the
character mismatch (CER), regenerating parts above 15% up to 3 times.

Voice match (cosine similarity of speaker embeddings, Chatterbox's own voice encoder):
  sim_generated_vs_input       generated speech vs the recording it copied (what the demo shows)
  sim_generated_vs_other_clip  generated speech vs a DIFFERENT real clip of the same speaker
                               (fairer: the model never saw that clip)
  sim_real_same_speaker        real clip vs that other real clip of the same speaker  (anchor)
  sim_real_different_speaker   real clip vs a real clip of a different speaker         (anchor)
Resumable: rows are saved after every run, and finished (clip, language) pairs are skipped.

Caveat: the encoder is the same one the voice model is conditioned on, so these scores
may flatter the generated audio. No human listening test was done.

What this does NOT measure: whether the translation is correct (we did not measure NLLB).
"""
import os
import sys
import time

import librosa
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import demo  # noqa: E402  (reuses the demo's exact pipeline; does not launch the UI)
import voice  # noqa: E402
from common import RESULTS, load_clip  # noqa: E402

LANGS = ["French", "Spanish", "Hindi"]
PER_GROUP = 2
SEED = 0


def other_clip(speaker, exclude):
    """A real clip of this speaker that is not `exclude` (longest one, for a stable embedding)."""
    c = demo.CLIPS[(demo.CLIPS.speaker == speaker) & (demo.CLIPS.role == "eval") & (demo.CLIPS.clip_id != exclude)]
    c = c[c.clip_id.apply(lambda x: os.path.exists(os.path.join(demo.CACHE, "audio", x + ".npy")))]
    return c.sort_values("duration", ascending=False).clip_id.iloc[0]


def main():
    side = pd.read_csv(os.path.join(RESULTS, "all_clips_side_by_side.csv"), keep_default_na=False)
    side = side[(side.ref_words >= 8) & (side.ref_words <= 25)]
    sample = side.groupby("group", group_keys=False).sample(PER_GROUP, random_state=SEED)
    speakers = sorted(sample.speaker.unique())
    out_path = os.path.join(RESULTS, "voice_check.csv")
    rows = pd.read_csv(out_path).to_dict("records") if os.path.exists(out_path) else []
    done = {(x["clip_id"], x["language"]) for x in rows}
    for r in sample.itertuples():
        if all((r.clip_id, t) in done for t in LANGS):
            continue
        ref_voice = demo.speaker_voice(r.clip_id, r.speaker)
        other = load_clip(other_clip(r.speaker, r.clip_id))
        diff_speaker = speakers[(speakers.index(r.speaker) + 1) % len(speakers)]
        diff = load_clip(other_clip(diff_speaker, ""))
        sim_same = voice.compare(load_clip(r.clip_id), other)
        sim_diff = voice.compare(load_clip(r.clip_id), diff)
        for target in LANGS:
            if (r.clip_id, target) in done:
                continue
            text = demo.translate(r.personal_mode, "en", target)
            t0 = time.time()
            sr, audio, report = voice.speak(ref_voice, text, demo.NAME_TO_CODE[target], demo.listen)
            secs = time.time() - t0
            o = voice.overall(report)
            gen16 = librosa.resample(audio.astype(np.float32), orig_sr=sr, target_sr=16000)
            rows.append(dict(group=r.group, speaker=r.speaker, clip_id=r.clip_id, language=target,
                             english=r.personal_mode, translation=text, heard=o["heard"],
                             cer_first_try_pct=100 * o["first_cer"], cer_final_pct=100 * o["cer"],
                             generations=o["attempts"], parts=o["chunks"],
                             still_above_15pct=any(c["cer"] > voice.MAX_CER for c in report),
                             sim_generated_vs_input=o["voice_similarity"],
                             sim_generated_vs_other_clip=voice.compare(gen16, other),
                             sim_real_same_speaker=sim_same, sim_real_different_speaker=sim_diff,
                             audio_seconds=len(audio) / sr, seconds_to_make=secs))
            pd.DataFrame(rows).round(4).to_csv(out_path, index=False)  # save after every run
            print(f"{r.clip_id} {target}: CER {rows[-1]['cer_final_pct']:.1f}% "
                  f"sim {rows[-1]['sim_generated_vs_input']:.2f} ({secs:.0f}s)", flush=True)
    d = pd.DataFrame(rows).round(4)

    per_clip = d.drop_duplicates("clip_id")
    summary = {
        "runs": len(d), "clips": d.clip_id.nunique(), "speakers": d.speaker.nunique(),
        "groups": d.group.nunique(), "languages": ", ".join(LANGS),
    }
    for lang in LANGS + ["All"]:
        x = d if lang == "All" else d[d.language == lang]
        summary[f"{lang}_cer_first_try_mean_pct"] = x.cer_first_try_pct.mean()
        summary[f"{lang}_cer_final_mean_pct"] = x.cer_final_pct.mean()
        summary[f"{lang}_cer_final_median_pct"] = x.cer_final_pct.median()
        summary[f"{lang}_runs_still_above_15pct"] = int(x.still_above_15pct.sum())
    summary.update({
        "generated_vs_input_similarity_mean": d.sim_generated_vs_input.mean(),
        "generated_vs_other_clip_similarity_mean": d.sim_generated_vs_other_clip.mean(),
        "real_same_speaker_similarity_mean": per_clip.sim_real_same_speaker.mean(),
        "real_different_speaker_similarity_mean": per_clip.sim_real_different_speaker.mean(),
        "runs_generated_closer_than_real_different_speaker":
            int((d.sim_generated_vs_other_clip > d.sim_real_different_speaker).sum()),
        "seconds_to_make_mean": d.seconds_to_make.mean(),
        "audio_seconds_mean": d.audio_seconds.mean(),
    })
    s = pd.DataFrame({"measure": list(summary), "value": list(summary.values())})
    s.to_csv(os.path.join(RESULTS, "voice_check_summary.csv"), index=False)
    print(s.to_string(index=False))


if __name__ == "__main__":
    main()
