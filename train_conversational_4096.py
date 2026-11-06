"""
Train Tri-Chamber Deep DHVN (V=4096) for Fluent Conversational English
======================================================================
Key Innovations:
1. Tri-Chamber Architecture (Acoustic -> Lexical Syntactic -> Propositional Discourse)
2. Constant 19.61 MB Working RAM Volume (Zero KV-Cache)
3. Topological Memory Crystallization ('Fluid Sleep' Cycles - Option 1)
4. Continuous Waveform Rhythm Ingestion (Option 2)
5. Active Inference Reflex Loop (Helmholtz Free Energy minimization)
"""

import os
import time
import math
import random
from typing import List
from word_pipeline import FluidWordTokenizer, DeepDHVNEngine, sample_word

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
CKPT_DIR = os.path.join(BASE_DIR, "checkpoints")
os.makedirs(CKPT_DIR, exist_ok=True)

CORPUS_PATH = os.path.join(DATA_DIR, "conversational_corpus_4096.txt")
VOCAB_PATH = os.path.join(DATA_DIR, "conversational_vocab_4096.json")
CKPT_PATH = os.path.join(CKPT_DIR, "conversational_dhvn_4096.bin")

def build_training_dataset(tokenizer: FluidWordTokenizer, max_seqs: int = 700) -> List[List[int]]:
    with open(CORPUS_PATH, "r", encoding="utf-8") as f:
        all_lines = [l.strip() for l in f if l.strip()]

    # Separate dialogue lines vs general knowledge lines
    dialogue_lines = [l for l in all_lines if "human:" in l and "fluid:" in l]
    knowledge_lines = [l for l in all_lines if "human:" not in l and len(l) > 25]

    dataset = []
    # Weight dialogues 4x to emphasize fluent conversational English
    for l in dialogue_lines:
        toks = tokenizer.tokenize(l)
        if 6 <= len(toks) <= 45:
            for _ in range(4):
                dataset.append(toks)

    random.seed(42)
    sample_k = random.sample(knowledge_lines, min(len(knowledge_lines), 200))
    for l in sample_k:
        toks = tokenizer.tokenize(l)
        if 8 <= len(toks) <= 40:
            dataset.append(toks)

    random.shuffle(dataset)
    if len(dataset) > max_seqs:
        dataset = dataset[:max_seqs]
    return dataset

def main():
    print("=" * 80)
    print(" TRI-CHAMBER DEEP DHVN (V=4096) CONVERSATIONAL ENGLISH TRAINING")
    print("=" * 80)

    tokenizer = FluidWordTokenizer(vocab_size=4096)
    tokenizer.load(VOCAB_PATH)
    print(f"[Tokenizer] Loaded {tokenizer.vocab_size} tokens from {os.path.basename(VOCAB_PATH)}.")

    dataset = build_training_dataset(tokenizer, max_seqs=700)
    total_tokens_dataset = sum(len(s) - 1 for s in dataset)
    print(f"[Dataset] Prepared {len(dataset)} sequences ({total_tokens_dataset:,} training tokens).")

    # Initialize 3-chamber model:
    # np1=512 (Acoustic), np2=512 (Lexical Syntactic), np3=512 (Discourse Intent), dh=256
    engine = DeepDHVNEngine(
        vocab_size=4096,
        np1=512, dim_c1=16,
        np2=512, dim_c2=16,
        np3=512, dim_c3=16,
        dim_hid=256
    )
    mem_mb = engine.get_memory_footprint() / (1024.0 * 1024.0)
    print(f"[Architecture] 3-Chamber Fluid Void (1,536 particles, Hidden Dim: 256).")
    print(f"[Architecture] Constant Working RAM: strictly {mem_mb:.2f} MB (Zero KV-Cache).")
    print("-" * 80)

    epochs = 18
    dt = 0.02
    initial_lr = 0.020
    min_lr = 0.006

    print(f"[Training] Running {epochs} epochs with Topological Sleep Cycles & Waveform Ingestion...")
    print(f"{'Epoch':<8} | {'NLL Loss':<10} | {'Perplexity':<12} | {'Throughput':<14} | {'Memory (MB)':<12} | {'Sleep Cycle'}")
    print("-" * 80)

    for ep in range(1, epochs + 1):
        # Cosine learning rate decay
        progress = (ep - 1) / max(1, epochs - 1)
        lr = min_lr + 0.5 * (initial_lr - min_lr) * (1.0 + math.cos(math.pi * progress))

        random.shuffle(dataset)
        total_loss = 0.0
        tokens_processed = 0
        t0 = time.time()

        for seq in dataset:
            in_seq = seq[:-1]
            tgt_seq = seq[1:]
            loss = engine.train_tokens(in_seq, tgt_seq, lr=lr, dt=dt)
            total_loss += loss * len(in_seq)
            tokens_processed += len(in_seq)

        t1 = time.time()
        avg_loss = total_loss / max(1, tokens_processed)
        ppl = math.exp(min(12.0, avg_loss))
        speed = tokens_processed / max(0.001, t1 - t0)

        # Option 1: Execute Topological Sleep / Crystallization every 3 epochs
        sleep_str = "None"
        if ep % 3 == 0:
            engine.crystallize_memory(sleep_cycles=6, cooling_rate=0.12)
            sleep_str = "Consolidated (6 cycles)"

        # Option 2: Periodic Continuous Waveform Influx (Prosodic cadence pulse)
        if ep % 6 == 0:
            prosody_pcm = [math.sin(2.0 * math.pi * 120.0 * (i / 16000.0)) * 0.2 for i in range(256)]
            engine.inject_waveform(prosody_pcm, dt=dt)
            sleep_str += " + Waveform"

        print(f"{ep:02d}/{epochs:02d}    | {avg_loss:8.4f}   | {ppl:10.2f}   | {speed:6.1f} tok/s   | {mem_mb:8.2f} MB  | {sleep_str}")

    # Final deep crystallization before checkpointing
    print("-" * 80)
    print("[Crystallization] Running final deep consolidation sleep (10 cycles)...")
    engine.crystallize_memory(sleep_cycles=10, cooling_rate=0.15)

    # Save trained checkpoint
    saved = engine.save_weights(CKPT_PATH)
    file_kb = os.path.getsize(CKPT_PATH) / 1024.0 if os.path.exists(CKPT_PATH) else 0.0
    print(f"[Checkpoint] Saved trained weights to {CKPT_PATH} ({file_kb:.1f} KB). Status: {saved}")
    print("=" * 80)

    # Interactive Verification Probes
    test_prompts = [
        "human: hello! how are you today? fluid:",
        "human: who are you? fluid:",
        "human: what is water? fluid:",
        "human: why is kindness important? fluid:",
        "human: what happens if temperature drops below freezing? fluid:"
    ]

    print("[Evaluation] Conversational Generation Test (Direct vs Active Inference 3 Ponder Steps):")
    print("=" * 80)

    for p in test_prompts:
        toks = tokenizer.tokenize(p)

        # 1. Direct Generation
        for t in toks:
            engine.predict_distribution(t, dt=dt)
        curr = toks[-1]
        gen_direct = list(toks)
        for _ in range(16):
            probs = engine.predict_distribution(curr, dt=dt)
            nxt = sample_word(probs, tokenizer, temperature=0.55, top_p=0.85, recent_tokens=gen_direct)
            gen_direct.append(nxt)
            curr = nxt
            if tokenizer.id2word.get(nxt, "") == ".":
                break
        direct_text = tokenizer.detokenize(gen_direct)

        # 2. Active Inference Pondered Generation
        for t in toks:
            engine.predict_distribution(t, dt=dt)
        curr = toks[-1]
        gen_ponder = list(toks)
        last_fe = []
        for _ in range(16):
            probs, fe = engine.active_inference_step(curr, ponder_steps=3, reflex_strength=0.35, dt=dt)
            last_fe = fe
            nxt = sample_word(probs, tokenizer, temperature=0.55, top_p=0.85, recent_tokens=gen_ponder)
            gen_ponder.append(nxt)
            curr = nxt
            if tokenizer.id2word.get(nxt, "") == ".":
                break
        ponder_text = tokenizer.detokenize(gen_ponder)
        fe_str = " -> ".join([f"{x:.3f}" for x in last_fe]) if last_fe else "N/A"

        print(f"Prompt  : {p}")
        print(f"Direct  : {direct_text}")
        print(f"Pondered: {ponder_text} [FE: {fe_str}]")
        print("-" * 80)

    engine.close()
    print("[Engine] Tri-chamber conversational English training completed successfully.")

if __name__ == "__main__":
    main()

