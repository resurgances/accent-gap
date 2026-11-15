"""
E4 (stretch): a short LoRA fine-tune of whisper-small on SPEAKER-DISJOINT data.

Question: if we fine-tune Whisper a little on accented conversational English from
OTHER people (the 32 DEV speakers: Irish, Scottish, Kenyan, South African, Italian,
Vietnamese), does it get better on our 40 evaluation speakers, and do accent gaps change?
None of the 40 evaluation speakers is used for training (speaker-disjoint), so there is
no leakage.

LoRA ("low-rank adaptation"): instead of changing all 240 million weights, we freeze
the model and train small extra matrices (here ~1.8M numbers) added to the attention
layers. It is cheap and quick, and the original model stays untouched.

Caveat we report: fine-tuning on EdAcc also teaches Whisper EdAcc's writing style
(e.g. writing every repeated "I I I"), so part of any gain is "learning the dataset's
style", not "understanding accents better".

Steps: download DEV speakers' audio (all their clips, 1-30 s) -> train 1 epoch ->
save adapter to lora_out/ (git-ignored). Then:
  python run_transcribe.py --model small-lora --mode plain
"""
import os
import random
import time

import numpy as np
import pandas as pd
import torch

from common import CACHE, ROOT, SAMPLE_RATE, audio_path, clean_reference, get_device, load_clip
from prepare_data import DUR_LOG, MAX_SEC, MIN_SEC, download_row_group

LORA_DIR = os.path.join(ROOT, "lora_out")
EPOCHS, BATCH, LR, SEED = 1, 8, 1e-3, 0
MAX_PER_SPEAKER = 40  # keeps training short and balanced across speakers


def training_clips():
    """All clips (history + dev) of the DEV speakers, with audio downloaded."""
    clips = pd.read_csv(os.path.join(CACHE, "clips.csv"))
    dev = clips[clips.group == "DEV"]
    assert not set(dev.speaker) & set(clips[clips.role == "eval"].speaker), "speaker overlap!"
    meta = pd.read_csv(os.path.join(CACHE, "metadata.csv")).set_index("clip_id")
    dev = dev.join(meta[["shard", "row_group"]], on="clip_id")
    done = pd.read_csv(DUR_LOG)
    todo = dev[~dev.clip_id.isin(done.clip_id)]
    print(f"DEV clips: {len(dev)}, audio still to download: {len(todo)}")
    for (shard, rg), d in todo.groupby(["shard", "row_group"]):
        res = download_row_group(shard, rg, set(d.clip_id))
        pd.DataFrame(res, columns=["clip_id", "duration"]).to_csv(DUR_LOG, mode="a", header=False, index=False)
    durs = pd.read_csv(DUR_LOG).drop_duplicates("clip_id").set_index("clip_id").duration
    dev = dev.assign(duration=dev.clip_id.map(durs))
    dev = dev[dev.duration.between(MIN_SEC, MAX_SEC) & dev.clip_id.apply(lambda c: os.path.exists(audio_path(c)))]
    # at most MAX_PER_SPEAKER clips per speaker (random but fixed by the seed)
    return dev.sample(frac=1, random_state=SEED).groupby("speaker").head(MAX_PER_SPEAKER)


def main():
    from peft import LoraConfig, get_peft_model
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    random.seed(SEED)
    torch.manual_seed(SEED)
    data = training_clips()
    print(f"training on {len(data)} clips from {data.speaker.nunique()} DEV speakers, "
          f"{data.duration.sum() / 60:.1f} minutes")

    processor = WhisperProcessor.from_pretrained("openai/whisper-small")
    processor.tokenizer.set_prefix_tokens(language="english", task="transcribe")
    model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")
    lora = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, target_modules=["q_proj", "v_proj"])
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()
    device = get_device()
    model.to(device).train()
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=LR)

    items = list(zip(data.clip_id, data.ref))
    steps_total = EPOCHS * (len(items) // BATCH)
    step, t0 = 0, time.time()
    for epoch in range(EPOCHS):
        random.shuffle(items)
        for i in range(0, len(items) - BATCH + 1, BATCH):
            batch = items[i:i + BATCH]
            audios = [load_clip(c) for c, _ in batch]
            # Target text: the cleaned EdAcc transcript in lowercase
            texts = [clean_reference(t).lower() for _, t in batch]
            feats = processor.feature_extractor(audios, sampling_rate=SAMPLE_RATE,
                                                return_tensors="pt").input_features.to(device)
            lab = processor.tokenizer(texts, return_tensors="pt", padding=True)
            labels = lab.input_ids.masked_fill(lab.attention_mask == 0, -100)
            # the model adds the start token itself, so drop it from the labels
            if (labels[:, 0] == model.config.decoder_start_token_id).all():
                labels = labels[:, 1:]
            loss = model(input_features=feats, labels=labels.to(device)).loss
            loss.backward()
            # linear warm-down of the learning rate
            for g in opt.param_groups:
                g["lr"] = LR * (1 - step / steps_total)
            opt.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            loss_value = loss.item()
            # free GPU memory every step (attempt 2 slowed down from memory build-up)
            del feats, labels, loss
            if device == "mps":
                torch.mps.empty_cache()
            if step % 20 == 0:
                mem = torch.mps.driver_allocated_memory() / 1e9 if device == "mps" else 0
                print(f"step {step}/{steps_total} loss {loss_value:.3f} "
                      f"({(time.time() - t0) / step:.2f} s/step, gpu mem {mem:.1f} GB)", flush=True)
    model.save_pretrained(LORA_DIR)
    print("saved adapter to", LORA_DIR)


if __name__ == "__main__":
    main()
