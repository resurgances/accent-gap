# LEARN.md: understand every part of Accent Gap

This file explains the project in plain language, phase by phase, so you can
present it live and answer questions from AI researchers. Every number here is
copied from a file in `results/` (the file name is given next to it).

---

## The big picture in one paragraph

Speech-to-text ("ASR", automatic speech recognition) turns audio into text.
We use **Whisper**, a free model from OpenAI that was trained on 680,000 hours
of internet audio. We ask two questions. (1) Does Whisper make more mistakes
for South Asian-accented English than for native US/UK English? (2) Can a cheap
"personal mode" help? It gives Whisper a short list of words that this person
often says, which is the kind of thing a phone already knows about its owner.
Then we check whether that closes the gap.

---

## Phase 0: setup and looking at the data

**What we built:** a Python environment, and `src/scan_metadata.py`. That script
reads only the *text columns* of the EdAcc dataset (speaker, accent, first
language, transcript) without downloading the 7 GB of audio. That works because
EdAcc is stored as Parquet files, where each column is stored separately.

**EdAcc** (Edinburgh International Accents of English Corpus) is a public dataset.
Pairs of people talk to each other over video calls in English. Speakers come
from many countries, and humans wrote down exactly what was said.

**What we found (`results/dataset_overview.csv`):** EdAcc has 19,137 clips, but
**very few South Asian speakers**:
- 5 speakers with an Indian English accent. One of them has only 10 usable clips, so we drop that speaker.
- 2 speakers whose first language is Urdu. They describe their own accent as "American".
- 2 speakers whose first language is Sinhalese (Sri Lanka).

That gives 8 South Asian speakers in total. This is a real limitation and we say
so everywhere. Our reference group has 15 native speakers: 6 US + 9 Southern British.

**Why measure speed first?** We timed one clip end to end. whisper-small took
about 0.2 s per clip on the MacBook's GPU ("MPS" = Apple's Metal GPU backend).
So compute was cheap and we could afford every usable clip instead of a small sample.

---

## Phase 1: choosing clips without cheating (`src/prepare_data.py`)

**Cleaning.** EdAcc transcripts contain tags like `<laugh>`, `<overlap>` and a
marker `IGNORE_TIME_SEGMENT_IN_SCORING`. These are not words anyone said, so we
remove them. We drop clips shorter than 1 s, longer than 30 s, or with empty
text. Most dropped clips are 1-word replies like "yeah" or "mm hmm". Whisper
only looks at 30 s at a time.

**The split, and why it matters (train/test split).** In machine learning you
must test on data the system never saw, or you are grading a student on the exact
questions they studied. Each EdAcc speaker is one conversation. We sort that
speaker's clips in time order:

- the **first 60%** = "history". This is like the messages your phone has already
  seen. We only use the *text* of these clips, to build the personal word list.
- the **last 40%** = "eval". Every experiment is scored **only** on these clips.

No Whisper run ever transcribes a history clip, and no word list ever sees eval
text. We also cap eval clips at 100 per speaker, so one very talkative person
cannot dominate a group's number.

---

## Key concepts (read these before presenting)

**WER (word error rate).** We line up Whisper's words against the human transcript.
Then we count substitutions (wrong word), deletions (missed word) and
insertions (extra word), and divide by the number of words actually said.
WER 10% means about 1 mistake every 10 words. Lower is better. It can go above
100% if the model invents lots of extra words.

**Normalization.** Before counting errors we run Whisper's official English
normalizer on BOTH texts. It lowercases, removes punctuation, writes numbers
the same way ("twenty" = "20"), makes US/UK spellings match ("colour" = "color"),
and drops fillers like "um". Without it, "Okay." vs "OKAY" would count as an error
even though nothing was misheard.

**Pooled WER.** For a group we add up all the errors and divide by all the
reference words. Longer clips count more. We also report the average of each
speaker's WER (`mean_speaker_wer_pct`).

**Baseline.** The simple thing you must beat. Here the baseline is plain Whisper
with no word list. For personal mode we add a second, tougher baseline, the
**control**: give Whisper *someone else's* word list (same accent group,
different conversation). If the control helps as much as the real list, then
the help comes from prompting in general, not from personalization.

**Prompting vs fine-tuning.**
- *Prompting*: we give the model extra text at the start, and Whisper treats it
  as "what was said just before". Nothing inside the model changes, it costs
  nothing, and it is instant.
- *Fine-tuning*: we keep training the model on new audio, so its internal
  numbers ("weights") change. That is more powerful, but it needs recorded audio of
  the user, takes compute, and can break other things.

Personal mode here is prompting only.

**Confidence interval (CI) via bootstrap.** If we had recorded 8 *different*
South Asian speakers, the number would change. The bootstrap estimates by how
much. We create 2,000 pretend datasets by picking speakers at random *with
replacement* (some twice, some not at all). We recompute WER each time, and the
middle 95% of those values is the 95% CI. We resample **speakers, not clips**,
because clips from the same person are not independent. With few speakers the
interval is wide, and that is the honest answer.

**Speaker-disjoint.** No person appears on both the training side and the testing
side. Our personal mode is *intentionally not* speaker-disjoint, because the
whole idea is to adapt to one person. Instead it is *time*-disjoint: earlier
conversation vs later conversation.
