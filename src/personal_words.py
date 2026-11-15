"""
Step 3: build each speaker's PERSONAL WORD LIST from their history clips only.

Idea ("mom mode"): your phone already knows things about you - names you text,
places you mention, words you use a lot. We simulate that with the reference
TEXT of the first 60% of each speaker's conversation (never the eval part).

Recipe (deliberately simple and explainable):
  1. take all words the speaker said in their history clips
  2. drop stopwords ("the", "and", "like", "yeah", ...) and very short words
  3. keep words said at least twice
  4. keep the 40 most frequent -> that's the list, joined with commas

We also make a CONTROL list for each speaker: the personal list of a different
speaker from the same accent group but a DIFFERENT conversation. If the control
helps as much as the real list, the gain comes from "prompting in general", not
from personalization.

Output: results/personal_word_lists.csv
"""
import os
import re
from collections import Counter

import pandas as pd

from common import CACHE, RESULTS

MAX_WORDS = 40
MIN_COUNT = 2

# Common English words + conversation fillers. These carry little information
# about a specific person, so they would just waste space in the prompt.
STOPWORDS = set("""
a about above after again against all also am an and any are aren't as at be because been
before being below between both but by can can't cannot could couldn't did didn't do does
doesn't doing don't down during each even ever every few for from further get gets getting
got gonna gotta had hadn't has hasn't have haven't having he he'd he'll he's her here here's
hers herself him himself his how how's i i'd i'll i'm i've if in into is isn't it it's its
itself just kind know let's like lot lots me mean more most much must mustn't my myself no
nor not now of off oh okay ok on once one only or other ought our ours ourselves out over own
pretty probably quite really right said same say says see shan't she she'd she'll she's should
shouldn't so some something sort still such sure than that that's the their theirs them
themselves then there there's these they they'd they'll they're they've thing things think
this those though through to too two under until up upon us very wanna was wasn't way we
we'd we'll we're we've well were weren't what what's when when's where where's whether which
while who who's whom why why's will with won't would wouldn't yeah yes yep you you'd you'll
you're you've your yours yourself yourselves um uh hmm mm mhm ah eh er actually basically
literally go going goes went come came make made take took want wanted good bad back time
also maybe many any anything everything nothing someone something somebody anyone people
guess feel felt three four five six seven eight nine ten first last new old big little
stuff yeah's alright cause 'cause gotta kinda dunno said tell told thought look looked
""".split())


def personal_list(texts):
    counts = Counter()
    for t in texts:
        for w in re.findall(r"[a-z']+", t.lower()):
            w = w.strip("'")
            if len(w) >= 3 and w not in STOPWORDS:
                counts[w] += 1
    common = [w for w, c in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])) if c >= MIN_COUNT]
    return common[:MAX_WORDS]


def main():
    clips = pd.read_csv(os.path.join(CACHE, "clips.csv"))
    rows = []
    for spk, d in clips.groupby("speaker"):
        hist = d[d.role == "history"]
        words = personal_list(hist.ref)
        rows.append({"speaker": spk, "group": d.group.iloc[0], "subgroup": d.subgroup.iloc[0],
                     "history_clips": len(hist), "n_words": len(words), "words": ", ".join(words)})
    lists = pd.DataFrame(rows).sort_values(["group", "speaker"]).reset_index(drop=True)

    # Control: pair each speaker with another speaker in the same group whose
    # conversation is different (conversation id = speaker name minus "-A"/"-B").
    conv = lists.speaker.str[:-2]
    control_spk = []
    for i, r in lists.iterrows():
        same = lists[(lists.group == r.group) & (conv != conv[i])]
        # rotate: pick the next eligible speaker after this one (deterministic)
        after = same[same.index > i]
        pick = after.iloc[0] if len(after) else same.iloc[0]
        control_spk.append(pick.speaker)
    lists["control_speaker"] = control_spk
    lists["control_words"] = [lists.set_index("speaker").loc[s, "words"] for s in control_spk]

    lists.to_csv(os.path.join(RESULTS, "personal_word_lists.csv"), index=False)
    for _, r in lists.iterrows():
        print(f"{r.speaker} [{r.subgroup}] ({r.n_words} words): {r.words[:120]}...")


if __name__ == "__main__":
    main()
