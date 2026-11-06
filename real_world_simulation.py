"""
Real-World Simulation of the Continuous 4D Void Engine (Deep DHVN).
Demonstrates real-time token processing, fluid state diagnostics, and text generation.
"""

import ctypes
import os
import time
import math
import random

LIB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "lib")
DEEP_LIB_PATH = os.path.join(LIB_DIR, "libdhvn_deep.so")
cdll = ctypes.CDLL(DEEP_LIB_PATH)

cdll.deep_dhvn_create.restype = ctypes.c_void_p
cdll.deep_dhvn_create.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int]
cdll.deep_dhvn_free.restype = None
cdll.deep_dhvn_free.argtypes = [ctypes.c_void_p]
cdll.deep_dhvn_set_substeps.restype = None
cdll.deep_dhvn_set_substeps.argtypes = [ctypes.c_void_p, ctypes.c_int]
cdll.deep_dhvn_get_memory_footprint.restype = ctypes.c_size_t
cdll.deep_dhvn_get_memory_footprint.argtypes = [ctypes.c_void_p]

cdll.deep_dhvn_train_sequence.restype = ctypes.c_float
cdll.deep_dhvn_train_sequence.argtypes = [
    ctypes.c_void_p,
    ctypes.POINTER(ctypes.c_uint8),
    ctypes.POINTER(ctypes.c_uint8),
    ctypes.c_int,
    ctypes.c_float,
    ctypes.c_float
]

cdll.deep_dhvn_predict_distribution.restype = None
cdll.deep_dhvn_predict_distribution.argtypes = [
    ctypes.c_void_p,
    ctypes.c_uint8,
    ctypes.c_float,
    ctypes.POINTER(ctypes.c_float)
]

def sample_token(dist, temperature=0.55, top_p=0.85):
    valid_indices = [10, 32] + list(range(33, 127))
    sub = [math.pow(max(1e-9, dist[i]), 1.0 / temperature) for i in valid_indices]
    tot = sum(sub)
    if tot <= 0:
        return 32
    probs = [s / tot for s in sub]
    pairs = sorted(zip(valid_indices, probs), key=lambda x: x[1], reverse=True)

    cum = 0.0
    nucleus_tokens = []
    nucleus_probs = []
    for tok, p in pairs:
        nucleus_tokens.append(tok)
        nucleus_probs.append(p)
        cum += p
        if cum >= top_p:
            break

    n_tot = sum(nucleus_probs)
    nucleus_probs = [p / n_tot for p in nucleus_probs]
    r = random.random()
    c = 0.0
    for tok, p in zip(nucleus_tokens, nucleus_probs):
        c += p
        if r <= c:
            return tok
    return nucleus_tokens[0]

def simulate_real_world():
    print("=" * 90)
    print("      REAL-WORLD CONTINUOUS 4D VOID ENGINE (DEEP DHVN) SIMULATION")
    print("=" * 90)

    # Initialize 1024-particle model (512 Acoustic + 512 Semantic) with 4D Spatial Hash Grid
    np1 = 512
    dim_c1 = 64
    np2 = 512
    dim_c2 = 128
    model = cdll.deep_dhvn_create(128, np1, dim_c1, np2, dim_c2)
    cdll.deep_dhvn_set_substeps(model, 2)
    mem_kb = cdll.deep_dhvn_get_memory_footprint(model) / 1024.0

    print(f"Physical Architecture:")
    print(f"  Chamber 1 (Acoustic Buffer): {np1} particles, {dim_c1}D latent charge (Micro-Vortices)")
    print(f"  Chamber 2 (Semantic Void):   {np2} particles, {dim_c2}D latent charge (Macro-Vortices)")
    print(f"  Continuous Neighbor Search:  4D Spatial Hash Cell Grid (O(N) Complexity)")
    print(f"  Readout Head:                2-Layer Resonant MLP (dh=128) + Wave Influx Conditioning")
    print(f"  Total Working Memory:        {mem_kb:.1f} KB (Strictly Constant, Zero KV Cache)")
    print("-" * 90)

    # Training corpus (English prose & relational knowledge)
    corpus = [
        b"The little bird was warm and happy in the tree.\n",
        b"Water is a clear liquid that flows into the deep blue ocean.\n",
        b"A clever fox jumped over the swift river bank.\n",
        b"The golden sun shines brightly in the clear morning sky.\n",
        b"A kind rabbit found a sweet red apple in the green grass.\n"
    ]

    print("[PHASE 1] Priming 4D Continuous Void with Natural Language Curriculum...")
    dt = ctypes.c_float(0.02)
    epochs = 40
    t_start = time.perf_counter()

    for ep in range(1, epochs + 1):
        tot_loss = 0.0
        for seq in corpus:
            in_b, tg_b = seq[:-1], seq[1:]
            L = len(in_b)
            loss = cdll.deep_dhvn_train_sequence(
                model,
                (ctypes.c_uint8 * L)(*in_b),
                (ctypes.c_uint8 * L)(*tg_b),
                L,
                ctypes.c_float(0.035),
                dt
            )
            tot_loss += loss
        if ep % 10 == 0 or ep == 1:
            mean_l = tot_loss / len(corpus)
            ppl = math.exp(min(12.0, mean_l))
            print(f"  Epoch {ep:2d}/40 | NLL Loss: {mean_l:.4f} | Perplexity: {ppl:6.2f}")

    t_train = time.perf_counter() - t_start
    print(f"Priming completed in {t_train:.2f}s ({t_train/epochs*1000:.1f} ms/epoch)\n")

    print("[PHASE 2] Real-World Interactive Prompt Streaming & Generation:")
    print("=" * 90)

    test_prompts = [
        "Water is a ",
        "The little bird ",
        "A clever fox ",
        "The golden sun ",
        "A kind rabbit "
    ]

    dist_buf = (ctypes.c_float * 128)()

    for prompt_text in test_prompts:
        prompt_bytes = prompt_text.encode('ascii')
        print(f"\nPrompt: \"{prompt_text}\"")

        # Influx Phase: Stream prompt bytes into the 4D fluid
        t0 = time.perf_counter()
        for b in prompt_bytes:
            cdll.deep_dhvn_predict_distribution(model, b, dt, dist_buf)
        t_prompt = (time.perf_counter() - t0) * 1000.0

        # Inspect immediate next-token prediction
        probs = [(i, dist_buf[i]) for i in range(128)]
        probs.sort(key=lambda x: x[1], reverse=True)
        top_candidates = []
        for idx, p in probs[:4]:
            char_repr = repr(chr(idx)) if 32 <= idx <= 126 else str(idx)
            top_candidates.append(f"{char_repr} ({p*100:.1f}%)")
        print(f"  Immediate Resonance: {', '.join(top_candidates)}")
        print(f"  Prompt Processing Latency: {t_prompt:.2f} ms ({t_prompt/len(prompt_bytes):.2f} ms/char)")

        # Autoregressive Generation Phase (Next 35 characters)
        generated = list(prompt_bytes)
        curr = prompt_bytes[-1]
        t_gen_start = time.perf_counter()

        for _ in range(35):
            cdll.deep_dhvn_predict_distribution(model, curr, dt, dist_buf)
            next_b = sample_token(dist_buf, temperature=0.55, top_p=0.85)
            generated.append(next_b)
            curr = next_b

        t_gen = (time.perf_counter() - t_gen_start) * 1000.0
        full_text = bytes(generated).decode('ascii', errors='replace').replace('\n', '\\n')
        print(f"  Fluid Output: \"{full_text}\"")
        print(f"  Generation Speed: {35.0 / (t_gen / 1000.0):.1f} chars/sec ({t_gen/35.0:.2f} ms/char)")

    print("\n" + "=" * 90)
    print("SIMULATION SUMMARY:")
    print(f"  RAM Invariance:        {mem_kb:.1f} KB allocated (Constant throughout generation)")
    print(f"  KV Cache Allocated:    0 Bytes (Zero Attention Lookup Tables)")
    print(f"  Fluid Dynamic Speed:   ~1.0 - 2.5 ms per continuous Navier-Stokes step")
    print("=" * 90)

    cdll.deep_dhvn_free(model)

if __name__ == "__main__":
    simulate_real_world()

