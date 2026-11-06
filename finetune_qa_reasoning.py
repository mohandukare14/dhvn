"""
Targeted Fine-Tuning of Deep DHVN (V=2048) on Question-Answering, Dialogue & Causal Deduction
===========================================================================================
Blends:
- 39 high-density QA and deductive reasoning chains from data/qa_reasoning_corpus.txt
- 50 interleaved anchor sequences from data/expanded_knowledge_corpus.txt (replay buffer)
Trains for 25 epochs using Adam on the 4D dual fluid chambers.
"""

import os
import time
import math
import random
from word_pipeline import FluidWordTokenizer, DeepDHVNEngine

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
CKPT_DIR = os.path.join(BASE_DIR, "checkpoints")

def main():
    print("=" * 80)
    print("   DEEP DHVN (V=2048) QA & CAUSAL REASONING FINE-TUNING PIPELINE")
    print("=" * 80)

    vocab_path = os.path.join(DATA_DIR, "word_vocab_2048.json")
    ckpt_path = os.path.join(CKPT_DIR, "word_dhvn_2048.bin")

    tokenizer = FluidWordTokenizer(vocab_size=2048)
    tokenizer.load(vocab_path)

    # 1. Load QA corpus
    qa_path = os.path.join(DATA_DIR, "qa_reasoning_corpus.txt")
    with open(qa_path, "r", encoding="utf-8") as f:
        qa_lines = [l.strip() for l in f if l.strip()]

    # 2. Load replay anchor sequences
    exp_path = os.path.join(DATA_DIR, "expanded_knowledge_corpus.txt")
    with open(exp_path, "r", encoding="utf-8") as f:
        exp_lines = [l.strip() for l in f if len(l.strip()) > 30]
    
    random.seed(42)
    replay_lines = random.sample(exp_lines, 50)

    # Combine: weight QA sequences 3x
    dataset = []
    for line in qa_lines:
        toks = tokenizer.tokenize(line)
        if len(toks) >= 4:
            for _ in range(3):
                dataset.append(toks)

    for line in replay_lines:
        toks = tokenizer.tokenize(line)
        if len(toks) >= 4:
            dataset.append(toks)

    random.shuffle(dataset)
    print(f"[Dataset] Prepared {len(dataset)} training sequences (QA weighted + replay buffer).")

    # Load existing trained model
    engine = DeepDHVNEngine(vocab_size=2048, np1=512, dim_c1=16, np2=512, dim_c2=16)
    if os.path.exists(ckpt_path):
        engine.load_weights(ckpt_path)
        print(f"[Checkpoint] Successfully loaded pre-trained weights from {ckpt_path}")
    else:
        print("[Warning] Checkpoint not found; starting fresh.")

    epochs = 20
    base_lr = 0.015
    dt = 0.02

    print(f"[Training] Running {epochs} fine-tuning epochs (LR={base_lr}, dt={dt})...")
    print("-" * 80)

    for ep in range(1, epochs + 1):
        random.shuffle(dataset)
        total_loss = 0.0
        total_tokens = 0
        t0 = time.time()

        for seq in dataset:
            in_seq = seq[:-1]
            tgt_seq = seq[1:]
            loss = engine.train_tokens(in_seq, tgt_seq, lr=base_lr, dt=dt)
            total_loss += loss * len(in_seq)
            total_tokens += len(in_seq)

        t1 = time.time()
        avg_loss = total_loss / max(1, total_tokens)
        ppl = math.exp(min(12.0, avg_loss))
        speed = total_tokens / max(0.001, t1 - t0)

        print(f"Epoch {ep:02d}/{epochs:02d} | NLL Loss: {avg_loss:6.4f} | PPL: {ppl:7.2f} | Speed: {speed:5.1f} tok/s | Elapsed: {t1-t0:4.1f}s")

    # Save fine-tuned weights
    engine.save_weights(ckpt_path)
    print("=" * 80)
    print(f"[Checkpoint] Fine-tuned model saved to {ckpt_path}")
    print("=" * 80)
    engine.close()

if __name__ == "__main__":
    main()
