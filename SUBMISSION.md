# Submission checklist (Veritas AI International High School Hackathon 2026)

Deadline: **Sunday Nov 15, 10:59 PM Central (11:59 PM ET)**. Aim to submit by 5 PM Central.

## Checklist
- [x] "The problem" in README.md: draft label removed (Claude wrote the first draft)
- [ ] Listen to the 6 clips in `notes/failure_notes.csv` (also marked in `results/worst_examples.csv`,
      column `listen_check_needed`; easiest way: the demo's "Real EdAcc examples" tab, clips marked ★).
      If an explanation is wrong, edit `notes/failure_notes.csv` and the README table, then re-run
      `.venv/bin/python src/analyze.py`
- [ ] Run the demo once yourself: `.venv/bin/python demo.py` (try your own voice too)
- [x] Record the 2-minute video using VIDEO_SCRIPT.md (screen-record the demo)
- [x] Upload the video and paste the link in README.md (https://youtu.be/ocqcZuP6GJk)
- [x] Publish the repo as PUBLIC on GitHub (done; still worth opening it once in a private window yourself)
- [x] Check that analysis.ipynb shows its outputs on GitHub (it loads publicly; open it once to look)
- [ ] Read LEARN.md fully and practise the judge questions out loud
- [ ] Fill in the form with the text below

## Copy-paste text for the form

**Project title:** Accent Gap: Does Speech-to-Text Understand South Asian Accents?

**Code link:** https://github.com/resurgances/accent-gap

**Write-up link:** https://github.com/resurgances/accent-gap#readme

**Write-up file:** `WRITEUP.pdf` (the README as a PDF, charts included):
https://github.com/resurgances/accent-gap/blob/main/WRITEUP.pdf

**Video:** https://youtu.be/ocqcZuP6GJk

**One-sentence pitch:** We measured whether OpenAI's Whisper understands South Asian-accented
English worse than native English on 1,628 real held-out clips, tested a "personal mode" that learns
each speaker's own words, and built a demo that transcribes your accent, translates it, and speaks
the translation back in your own voice.

**Anything else they should know:**
- Honest results: we expected a South Asian gap and did not find one on this dataset (20.6% vs
  23.1% word error rate, 95% CI of the ratio 0.69 to 1.20). Jamaican and Nigerian English were
  clearly worse. Personal mode lowered errors by 1.3 points overall, and the speaker's own words
  beat someone else's by 1.8 points (95% CI +0.5 to +3.2). Our LoRA fine-tune made things worse,
  and we report that too.
- Every number in the write-up comes from a file in `results/` produced by code in the repo. Word
  lists only use the earlier part of each conversation, and scoring uses the later part (no leakage).
- Everything runs locally on a laptop (no paid APIs). Public data only (EdAcc); no audio is in the repo.
- Main limitation: only 8 South Asian speakers in EdAcc and no Sindhi speakers, so "no gap found"
  is not "no gap exists". Next step: consented recordings of speakers like my parents.
- The voice-cloned audio is labelled AI-generated and watermarked by the model.
- Claude Code (Anthropic) was used as an AI coding assistant, and all tools and data are credited
  in the README.

**Project name:** Accent Gap

**Track:** Languages (AI + Languages)

**Team:** Karan Daryanani

**One-line description:** We measure how much worse Whisper speech-to-text is for South Asian-accented
English than for native US/UK English on public data, find which words it gets wrong, and test a
"personal mode" that gives Whisper a short list of words the speaker often uses.

**What we added (beyond the suggested "speech recognition across accents" project):**
- Our own user: South Asian-accented speakers dictating everyday messages, motivated by my mom, whose
  texts Siri dictation mangles while it types mine perfectly.
- Our own question: does per-speaker personalization (a personal word list built only from a
  speaker's earlier speech, passed to Whisper as a prompt) close the accent gap?
- A leak-free design: word lists use only the first 60% of each conversation; all scoring is on the last 40%.
- A control baseline (someone else's word list) to separate "personalization" from "prompting in general".
- Speaker-level bootstrap confidence intervals and per-speaker results, plus error analysis of which words fail.

**Credits:**
- Whisper speech recognition models (openai/whisper-base, -small, -medium), OpenAI: https://github.com/openai/whisper , https://huggingface.co/openai/whisper-small
- EdAcc, the Edinburgh International Accents of English Corpus (Sanabria et al., 2023): https://huggingface.co/datasets/edinburghcstr/edacc , paper: https://arxiv.org/abs/2303.18110
- NLLB-200 translation model (facebook/nllb-200-distilled-600M), Meta AI, used in the demo for translation (CC BY-NC 4.0): https://huggingface.co/facebook/nllb-200-distilled-600M
- Chatterbox Multilingual voice-cloning text-to-speech (chatterbox-tts 0.1.7, ResembleAI/chatterbox), Resemble AI, used in the demo to speak the translation in the speaker's voice (MIT license): https://github.com/resemble-ai/chatterbox
- Hugging Face transformers, datasets, huggingface_hub: https://github.com/huggingface/transformers
- PyTorch: https://pytorch.org
- jiwer (WER alignment): https://github.com/jitsi/jiwer
- librosa, soundfile, pyarrow, pandas, numpy, matplotlib, wordfreq, peft (LoRA), Gradio, uv, Jupyter
- Claude Code (Anthropic) was used as an AI coding assistant: https://www.anthropic.com/claude-code

## Publishing the repo on GitHub (done: https://github.com/resurgances/accent-gap; steps kept for reference)

1. In your browser go to https://github.com/new, name it `accent-gap`, choose **Public**, and do NOT
   add a README or .gitignore (the repo already has them). Click "Create repository".
2. In Terminal:

```bash
cd ~/Documents/accent-gap
git remote add origin https://github.com/YOUR-USERNAME/accent-gap.git
git push -u origin main
```

3. When git asks for a password, paste a **personal access token**, not your GitHub password.
   Create one at https://github.com/settings/tokens → "Generate new token (classic)" → tick `repo` → copy it.
   macOS Keychain will remember it.
4. Open the repo in a private window and check that README charts and analysis.ipynb outputs show.

**Links:**
- GitHub repo: https://github.com/resurgances/accent-gap
- Video: https://youtu.be/ocqcZuP6GJk
