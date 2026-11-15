# 2-minute video script: Accent Gap

Record with the screen showing the demo (`.venv/bin/python demo.py`) and the README charts.
Times are targets. Read it in your own voice and change the words so they sound like you.

---

### 0:00–0:15 · The problem (15 s)
**On screen:** your face, or your phone showing a dictated text.

> "My parents speak Sindhi. When I dictate a text, Siri types it perfectly. When my mom
> dictates on the same phone, it's a mess. Same phone, different accent. So I asked: how much
> worse is speech-to-text for South Asian accents, and can a phone adapt to one person?"

### 0:15–1:15 · Demo (60 s)
**On screen:** the Gradio demo.

1. **(0:15)** Tab "Speak", with "Translate to" set to **Sindhi**. Record yourself, for example
   "Tell Ammi I'll pick up the biryani from Karachi Kitchen at six", and press stop.
   > "I just talk. It detects that I'm speaking English, writes down what I said, and translates
   > it into Sindhi, my parents' language. I don't type anything."
2. **(0:30)** Record 2–3 more sentences, then point at "Words it has learned from you so far".
   > "The more I talk, the more of my words it learns. It gives them to the AI, OpenAI's Whisper,
   > as a hint for next time. That's my personal mode."
3. **(0:40)** Tab "Real EdAcc examples". Pick **★ EDACC-C10-84** (Sinhalese L1), press play, then
   "Run Standard vs With learned words".
   > "On a real clip from the public EdAcc dataset, standard Whisper writes 'No, contrary to that'.
   > With words learned from earlier in that conversation, it gets 'No, I can't remember' exactly
   > right. Mistakes are highlighted."
4. **(0:55)** Pick **★ EDACC-C52-312** (Jamaican: "me prefers Naruto more than Dragon Ball Z").
   > "And here's where it still fails: rare names plus a different grammar, and Whisper turns it
   > into a normal-sounding sentence that's wrong. It's a prototype, not a product."

### 1:15–1:45 · How it works + baseline (30 s)
**On screen:** README chart `results/wer_by_group.png`, then `results/personal_mode.png`.

> "I measured word error rate on 1,628 held-out clips from 40 speakers. The surprise: on this
> dataset Whisper was *not* worse for South Asian speakers, 20.6% vs 23.1% for native US and UK
> speakers. Jamaican and Nigerian English were clearly worse. Personal mode lowered errors
> by 1.3 points overall. A control using *someone else's* word list did nothing, so the gain
> really comes from personal words. My first version actually made things worse, because
> Whisper got stuck in loops, and I report that too."

### 1:45–2:00 · Limitation + next step (15 s)
**On screen:** README "Limitations".

> "The big limitation: only 8 South Asian speakers, all fluent and on video calls. No Sindhi
> speakers, and not my mom. Next, I want to record, with consent, older speakers like my
> parents dictating real messages, and test whether a small voice fine-tune helps more than a
> word list."

---
Credits slide (optional, 3 s): "Whisper (OpenAI) · NLLB-200 (Meta) · EdAcc (University of Edinburgh) · Hugging Face · Claude Code (Anthropic) as coding assistant"
