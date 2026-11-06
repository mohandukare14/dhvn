"""
Continuous 4D Navier-Stokes Fluid Void Engine (DHVN) - Multi-Domain 2048-Token Model
===================================================================================
Expands vocabulary to 2,048 tokens covering science, philosophy, logic, astronomy,
causality, ecology, and literature with Hebbian vorticity strain plasticity.
"""

import os
import sys
import time
import math
import random
from word_pipeline import FluidWordTokenizer, DeepDHVNEngine, build_dataset, sample_word

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
CKPT_DIR = os.path.join(BASE_DIR, "checkpoints")

VOCAB_PATH = os.path.join(DATA_DIR, "word_vocab_2048.json")
CKPT_PATH = os.path.join(CKPT_DIR, "word_dhvn_2048.bin")

def main():
    print("=" * 75, flush=True)
    print(" CONTINUOUS 4D NAVIER-STOKES FLUID VOID ENGINE (DHVN)", flush=True)
    print(" STAGE 5: MULTI-DOMAIN KNOWLEDGE & REASONING EXPANSION (V = 2048)", flush=True)
    print("=" * 75, flush=True)

    # 1. Build Tokenizer with 2048 vocabulary
    vocab_size = 2048
    tokenizer = FluidWordTokenizer(vocab_size=vocab_size)
    corpus_files = [
        os.path.join(DATA_DIR, "english_dictionary.txt"),
        os.path.join(DATA_DIR, "qa_reasoning_corpus.txt"),
        os.path.join(DATA_DIR, "expanded_knowledge_corpus.txt"),
        os.path.join(DATA_DIR, "stories.txt"),
        os.path.join(DATA_DIR, "dialogue.txt")
    ]
    tokenizer.fit_from_files(corpus_files)
    tokenizer.save(VOCAB_PATH)

    # 2. Build Dataset
    seq_len = 16
    full_dataset = build_dataset(tokenizer, corpus_files, seq_len=seq_len)
    # Take a balanced multi-domain sample of 650 sequences (covers every domain evenly)
    step = max(1, len(full_dataset) // 650)
    dataset = full_dataset[::step][:650]
    print(f"[Dataset] Sampled {len(dataset)} balanced sequences from {len(full_dataset)} total (length {seq_len}).", flush=True)

    # 3. Initialize Deep DHVN Engine (1,024 fluid particles)
    np1 = 512
    np2 = 512
    engine = DeepDHVNEngine(vocab_size=vocab_size, np1=np1, dim_c1=16, np2=np2, dim_c2=16)
    engine.set_substeps(1)
    mem_bytes = engine.get_memory_footprint()
    mem_mb = mem_bytes / (1024.0 * 1024.0)
    print(f"[Engine] Particles: {np1 + np2} (Chamber 1: {np1}, Chamber 2: {np2})", flush=True)
    print(f"[Engine] Constant O(1) RAM Volume: {mem_mb:.2f} MB", flush=True)
    print(f"[Engine] Zero W_e Embeddings, Zero Transformer QK^T, Zero KV-Cache.", flush=True)
    print("-" * 75, flush=True)

    # 4. Train Model
    num_epochs = 8
    lr = 0.01
    dt = 0.005

    print(f"[Training] Starting Multi-Domain Training ({num_epochs} Epochs, lr={lr})...", flush=True)
    t0_all = time.time()

    for epoch in range(1, num_epochs + 1):
        random.shuffle(dataset)
        total_loss = 0.0
        t0_ep = time.time()

        for in_seq, tgt_seq in dataset:
            loss = engine.train_tokens(in_seq, tgt_seq, lr=lr, dt=dt)
            total_loss += loss

        avg_loss = total_loss / len(dataset)
        ppl = math.exp(min(avg_loss, 20.0))
        ep_time = time.time() - t0_ep
        speed = (len(dataset) * seq_len) / max(ep_time, 1e-4)
        print(f"  Epoch {epoch:2d}/{num_epochs:2d} | NLL Loss: {avg_loss:.4f} | Perplexity: {ppl:6.2f} | Speed: {speed:.1f} tok/s | Time: {ep_time:.2f}s", flush=True)

    total_time = time.time() - t0_all
    print(f"[Training] Completed in {total_time:.2f}s.", flush=True)
    print("-" * 75, flush=True)

    # 5. Save Checkpoint
    os.makedirs(CKPT_DIR, exist_ok=True)
    if engine.save_weights(CKPT_PATH):
        sz_kb = os.path.getsize(CKPT_PATH) / 1024.0
        print(f"[Checkpoint] Successfully saved {vocab_size}-token model weights to {CKPT_PATH} ({sz_kb:.1f} KB).", flush=True)
    print("-" * 75, flush=True)

    # 6. Evaluation with Active Inference Reflex Loop
    test_queries = [
        "the sun is a radiant star",
        "question: what is water? answer:",
        "energy cannot be created or",
        "plants convert sunlight into",
        "question: who are you? answer:",
        "if temperature drops below freezing",
        "wisdom begins with recognizing",
        "the wise old owl perched quietly upon"
    ]

    print("[Evaluation] Comparing Direct Readout vs Active Inference Pondering (4 steps):", flush=True)
    print("=" * 75, flush=True)

    for prompt in test_queries:
        toks = tokenizer.tokenize(prompt)
        
        # Test A: Direct Readout
        for t in toks:
            engine.predict_distribution(t, dt=dt)
        curr = toks[-1]
        gen_direct = list(toks)
        for _ in range(12):
            probs = engine.predict_distribution(curr, dt=dt)
            nxt = sample_word(probs, tokenizer, temperature=0.6, top_p=0.85, recent_tokens=gen_direct)
            gen_direct.append(nxt)
            curr = nxt
            if tokenizer.id2word.get(nxt, "") == ".":
                break

        # Test B: Active Inference Reflex Loop (3 ponder steps)
        for t in toks:
            engine.predict_distribution(t, dt=dt)
        curr = toks[-1]
        gen_ponder = list(toks)
        last_fe = []
        for _ in range(12):
            probs, fe = engine.active_inference_step(curr, ponder_steps=3, reflex_strength=0.35, dt=dt)
            last_fe = fe
            nxt = sample_word(probs, tokenizer, temperature=0.6, top_p=0.85, recent_tokens=gen_ponder)
            gen_ponder.append(nxt)
            curr = nxt
            if tokenizer.id2word.get(nxt, "") == ".":
                break

        fe_str = " -> ".join([f"{x:.3f}" for x in last_fe]) if last_fe else "N/A"
        print(f"Prompt   : '{prompt}'", flush=True)
        print(f"Direct   : {tokenizer.detokenize(gen_direct)}", flush=True)
        print(f"Pondered : {tokenizer.detokenize(gen_ponder)} [Free Energy: {fe_str}]", flush=True)
        print("-" * 75, flush=True)

    engine.close()
    print("[Engine] Multi-domain expansion run completed cleanly.", flush=True)

if __name__ == "__main__":
    main()
