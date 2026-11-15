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

---

## Phase 2: a bug we caught (and why it matters)

Some EdAcc conversations were recorded in parts (clip ids like `EDACC-C31_P1-12`
and `EDACC-C31_P2-3`), and **clip numbers restart in every part**. Our first
"conversation order" sorted by clip number alone, which mixed parts together,
so some "history" clips actually came *after* "eval" clips. We found this while
reading clip ids during error analysis, fixed the sort to (part, clip number),
and re-ran everything. The results in the README are all from the fixed version.
Lesson: always look at your raw data; a split that is *almost* right can still leak.

## Phase 3: personal mode v1 failed, and we fixed it without cheating

**What happened.** Our first personal mode (v1) was a 40-word comma-separated
list passed as the Whisper prompt, and it made WER *worse* in every group. When we
looked at the worst clips, the cause was clear: on short replies like
"no no no", Whisper copied the list format and got stuck in a **repetition loop**
("no, no, no, no, ..." about 100 times until it hit the length limit). Each loop
adds about 100 "inserted" words, so a handful of clips wiped out the small gains
on hundreds of other clips.

**Why we could not just try fixes on the test clips.** If you try 8 versions and
keep the one that scores best on the test set, the test score is no longer
honest. You have partly fitted the test, which is a form of leakage. So we made
a **dev set** (development set): 32 other EdAcc speakers (Irish, Scottish,
Kenyan, South African, Italian, Vietnamese) who are in none of our result groups.
We tried the variants there, picked the winner with a rule fixed in advance
(lowest dev WER), and ran it **once** on the evaluation clips. We still report
v1's evaluation result, because it is a real negative result.

**The loop guard.** The original Whisper code detects loops by compressing the
output text with zlib. Normal sentences barely shrink, but "no, no, no, ..."
shrinks a lot. If the compression ratio is above 2.4, Whisper's own threshold,
the output is treated as a failed decode. Our guard says: if personal-mode output
looks like a loop, use the standard output instead. On a phone that means "re-run
without the word list".

---

## Phase 4: the results, explained

All numbers are whisper-small on the 1,628 held-out eval clips (40 speakers) unless stated.

### E1: is there an accent gap? (`results/wer_by_group.csv`, `results/accent_gap.csv`)
- South Asian English: **20.6%** WER (95% CI 17.1-26.2), 8 speakers
- Native US/UK English: **23.1%** (19.2-27.5), 15 speakers
- Nigerian 28.1%, Spanish-accented 25.4%, Jamaican 37.6% (CI 24.4-58.9, very wide)
- Ratio South Asian / native = **0.89** (CI 0.69-1.20). The CI includes 1.0, so **we found no
  gap** for South Asian speakers on EdAcc. Jamaican (1.63x) and Nigerian (1.21x) speakers do
  show gaps whose CIs stay above 1.

**How to say this to judges:** "My hypothesis was wrong for this dataset, and I'm reporting
that. EdAcc's South Asian speakers are fluent young speakers on video calls, so it tests a
different situation than my mom dictating on a phone."

**Speakers differ more than groups do.** South Asian speakers range from 10.1% to 34.8% WER,
native speakers from 3.3% to 39.1% (`results/wer_by_speaker.csv`). One US speaker (C07-A) has
high WER because Whisper returned almost nothing for several of their clips.

### E2: does a bigger model help? (`results/wer_by_group_all_systems.csv`)
WER for base → small → medium: South Asian 25.7 → 20.6 → 18.0; native 29.0 → 23.1 → 18.7.
Bigger is better for everyone. But for Nigerian and Spanish-accented speakers the *ratio* to
native gets worse with medium (1.42x and 1.36x), because native speakers gain more. A bigger
model is not a fix for accent gaps.

### E3: personal mode (`results/personal_mode_by_group.csv`, `results/personal_vs_control.csv`)
- Final design (sentence prompt, 40 words, loop guard): all 40 speakers go from 25.5% → 24.3%,
  a gain of **1.3 points** (CI -0.1 to +2.7). It is a small gain and only borderline significant.
- South Asian 20.6 → 19.3 (+1.3); native 23.1 → 21.9 (+1.2). The difference between those two
  gains is +0.1 (CI -4.0 to +4.5), so there is **no evidence it helps South Asian speakers more**.
- **Control** (someone else's list): 26.1% overall, slightly *worse* than no prompt. The own
  list beats the control by **1.8 points (CI +0.5 to +3.2)**. This is our most solid finding:
  *personal* words help, while random prompts do not.
- v1 (a bare comma list) made things worse because of repetition loops (see Phase 3).
- **Model size matters.** The same final design helps whisper-base (+1.4) and whisper-small (+1.3),
  but **hurts whisper-medium** (-2.7, CI -3.9 to -1.3). With a prompt, medium often stops after
  one word ("I", "Yeah.") on long clips, so deletions go up (South Asian 11.7% -> 15.0%). We
  chose the design with whisper-small on dev speakers, and it did not transfer. Lesson: a
  prompt that works for one model can break another, so you must re-test per model.

### E4 (stretch): LoRA fine-tune (`results/lora_vs_plain.csv`)
**LoRA** = "low-rank adaptation". Instead of retraining all ~240 million weights, we freeze
Whisper and train small extra matrices (about 1.8 million numbers, 0.7%) inside the attention
layers. We trained only on the 32 DEV speakers (speaker-disjoint: none of the 40 test speakers).
Result: WER got **worse**, 25.5% -> 31.2% over all 40 speakers (-5.7, CI -11.1 to -0.4). The model
learned to write repeated words like EdAcc does (deletions 13.9% -> 5.3%), but it also learned to
loop ("they they they ..."): clips with 5+ inserted words went from 17 to 56. Lesson: fine-tuning
is powerful but easy to get wrong, and a quick untuned run is not a fair test of the idea. Our first
two training attempts were also too slow (GPU shared with another job, then a GPU memory
build-up), and the third, fixed run is the one reported.

### Error analysis (`results/miss_rate_by_word_type.csv`, `results/wer_by_length.csv`, `results/missed_words_by_group.csv`)
- Rare words (names, places) are missed about 2-3x as often as common words (South Asian
  49.5% vs 18.3%).
- Most errors are **deletions** of small words ("i", "you", "like", "know") in fast,
  overlapping conversation. Whisper writes clean sentences, while EdAcc writes every "I I I".
- Short clips (1-3 s) are much worse. At 10-30 s, every group is between 19.2% and 24.0%.
- Real failure examples, with our explanations: README section "Where it fails" and
  `results/worst_examples.csv` (column `listen_check_needed` = you should listen to confirm).

---

## Phase 5: the demo (`demo.py`)

The demo uses **three AI steps**, all running on the laptop:
1. **Language detection (Whisper).** Before transcribing, Whisper guesses which of its ~100
   languages is being spoken. It does this by predicting the "language token" (like `<|en|>`)
   that normally starts its output.
2. **Speech-to-text (Whisper)**, plus *learn my words*. Every time you speak, the demo saves what
   you said. Next time, it runs the exact recipe from `personal_words.py` on your earlier speech:
   drop stopwords, keep words you said at least twice, top 40. The list is passed to Whisper as a
   prompt, with the loop guard. It is only used for English with whisper-small, because that is
   where our experiments showed it helps. On whisper-medium it hurt, so the demo switches it off there.
3. **Translation (NLLB-200, Meta).** NLLB is a separate model trained to translate between 200
   languages, including Sindhi (`snd_Arab`). We translate one sentence at a time, because when we
   tested it on two sentences at once it dropped the first sentence in French.

**Important honesty point:** we measured speech-to-text accuracy, not translation quality or
language-detection accuracy. In the demo those are prototype features, and the README says so.
Also, the learned words come from the model's own earlier output, so if it mishears a word twice,
it may "learn" the wrong word. A real phone would learn from texts you corrected.

---

## Likely judge questions (with honest answers)

**1. "Your hypothesis was that Whisper is worse for South Asian accents. Did you prove it?"**
No. On EdAcc, South Asian speakers had 20.6% WER vs 23.1% for native speakers, and the
confidence interval of the ratio (0.69-1.20) includes "no difference". With only 8 South Asian
speakers we can't rule out a gap either. What we *can* say is that on fluent, conversational,
video-call English, Whisper-small did not show a large South Asian gap, while Jamaican and
Nigerian English did.

**2. "Then why does Siri fail for your mom?"**
I don't know for sure, and my data can't answer it. Possible reasons: Siri is a different
model; my mom is an older speaker whose first language is Sindhi, while EdAcc's South Asian speakers
are young and fluent; and dictation on a phone is a different setting. Testing that would need
consented recordings from speakers like her, which is my next step.

**3. "Why resample speakers for the confidence interval, not clips?"**
Clips from one person share the same voice, microphone and topic, so they are not independent.
If I resampled clips, I'd pretend I had 356 independent South Asian data points when I really
have 8 people, and the interval would be falsely narrow.

**4. "How do you know you didn't leak test data into personal mode?"**
Each speaker's conversation is split in time order. The word list uses only the text of the
first 60%, and scoring uses only the last 40%. No model ever transcribes history clips.
Design choices (prompt format, loop guard) were chosen on 32 *different* dev speakers. I also
found and fixed a bug where conversation parts were mis-ordered, and re-ran everything.

**5. "Isn't a 1.3-point gain tiny?"**
Yes, it is about 5% fewer errors, and its CI just touches zero. The stronger result is the
comparison with someone else's list: own list beats the control by 1.8 points (CI +0.5 to
+3.2). So the idea works a bit, but a word list is a weak form of personalization.

**6. "What's the difference between prompting and fine-tuning?"**
Prompting gives the model extra text at the start ("words I often use: ..."), and the model's
weights don't change. Fine-tuning keeps training the model on new audio so the weights change.
Fine-tuning could learn how a person *pronounces* words, but it needs their recordings, compute,
and care to avoid breaking the model. Prompting is free and instant but can only hint which words
to expect.

**7. "Why did version 1 make things worse?"**
A bare comma list ("cool, nice, food, ...") made Whisper continue in the same pattern, so on
short replies it got stuck repeating ("no, no, no, ..." about 100 times). Each loop adds
about 100 wrong words. We fixed it with Whisper's own loop detector (zlib compression ratio
> 2.4 → fall back to normal output) and a sentence-style prompt, chosen on dev speakers.

**8. "What is WER and why normalize?"**
WER = (wrong + missing + extra words) / words actually said. We normalize both texts with
Whisper's English normalizer (lowercase, no punctuation, "twenty" = "20", "colour" = "color",
drop "um") so formatting differences don't count as recognition errors.

**9. "What's your baseline?"**
Plain Whisper with no prompt, on exactly the same clips. For personal mode there's a second
baseline: the same prompt format with someone else's word list (the control).

**10. "Would personal mode work on a bigger, better model?"**
Not as we built it. On whisper-medium the same prompt made results worse (-2.7 points),
because medium tends to stop transcribing early when given a prompt. Personal words still did
better than someone else's words, but both were worse than no prompt. A product would need to
tune the prompt for each model, or use a different method (like fine-tuning).

**11. "How does the demo translate? Is the translation accurate?"**
Whisper first writes down the speech in the language you spoke, then Meta's NLLB-200 model
translates the text. We did not measure translation quality, so I can't claim it's accurate.
I only checked a few sentences by eye. Errors can stack: if Whisper mishears a word, the
translation will be wrong too.

**12. "What's the AI here, did you just call an API?"**
The AI is OpenAI's Whisper, a neural network that turns audio into text. It runs locally on my
laptop (no API). My contributions are the evaluation design (groups, leak-free split, bootstrap
CIs, control, dev set), the personal-mode method and loop guard, the error analysis, and the
demo. I used Claude Code as a coding assistant, and the README credits it.
