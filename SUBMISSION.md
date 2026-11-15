# Submission checklist (Veritas AI International High School Hackathon 2026)

Deadline: **Sunday Nov 15, 10:59 PM Central (11:59 PM ET)**. Aim to submit by 5 PM Central.

## Checklist
- [ ] Rewrite "The problem" in README.md in your own words (Claude wrote a draft)
- [ ] Listen to the 5 clips marked in `results/worst_examples.csv` (column `listen_check_needed`)
      and fix the "why we think it failed" explanations if they are wrong
- [ ] Run the demo once yourself: `.venv/bin/python demo.py` (try your own voice too)
- [ ] Record the 2-minute video using VIDEO_SCRIPT.md (screen-record the demo)
- [ ] Upload the video (YouTube unlisted or Google Drive "anyone with link") and paste the link in README.md
- [ ] Publish the repo as PUBLIC on GitHub and open it in a private/incognito window to check it is visible
- [ ] Check that analysis.ipynb shows its outputs on GitHub
- [ ] Read LEARN.md fully and practise the judge questions out loud
- [ ] Fill in the form with the text below

## Copy-paste text for the form

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
- Hugging Face transformers, datasets, huggingface_hub: https://github.com/huggingface/transformers
- PyTorch: https://pytorch.org
- jiwer (WER alignment): https://github.com/jitsi/jiwer
- librosa, soundfile, pyarrow, pandas, numpy, matplotlib, wordfreq, Gradio, Jupyter
- Claude Code (Anthropic) was used as an AI coding assistant: https://www.anthropic.com/claude-code

**Links:**
- GitHub repo: (paste after publishing)
- Video: (paste after uploading)
