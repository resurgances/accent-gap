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

1. **(0:15)** Tab "Real EdAcc examples". Pick the **Sinhalese L1 / EDACC-C10-84** clip and press play.
   > "This is a real clip from a public dataset of accented English, EdAcc. The speaker says
   > 'No, I can't remember'."
2. **(0:25)** Show the word list box.
   > "Personal mode gives the AI, OpenAI's Whisper, a short list of words this person used
   > *earlier* in the conversation. Like what your phone already knows about you."
3. **(0:32)** Press "Run Standard vs Personal mode".
   > "Standard Whisper writes 'No, contrary to that'. Personal mode gets it exactly right.
   > Mistakes are highlighted."
4. **(0:45)** Pick the **Jamaican / EDACC-C52-312** clip ("me prefers Naruto more than Dragon Ball Z").
   > "And here's where both fail: rare names plus a different grammar, and Whisper turns it
   > into a normal-sounding sentence that's wrong."
5. **(0:58)** Tab "Try it", record yourself saying one sentence with a family name or a place
   (for example "Tell Ammi I'll pick up the biryani from Karachi Kitchen at six"), and add
   those names to the word list.
   > "You can try it with your own voice too. It's a prototype, not a product."

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
Credits slide (optional, 3 s): "Whisper (OpenAI) · EdAcc (University of Edinburgh) · Hugging Face · Claude Code (Anthropic) as coding assistant"
