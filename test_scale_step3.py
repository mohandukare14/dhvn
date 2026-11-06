"""
Step 3: Scaled Parameter Capacity Benchmark.
Scale c1 = 64, c2 = 128, dh = 256 to break toward Loss < 1.50.
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

curriculum = [
    b"The little bird was warm and happy.\n",
    b"Water is a clear liquid that flows into the ocean.\n",
    b"A clever fox jumped over the river bank.\n",
    b"The golden sun shines brightly in the sky.\n",
    b"A kind rabbit found a sweet red apple.\n"
]

def run_scaled_trial():
    print("=" * 80)
    print("STEP 3: SCALED CAPACITY BENCHMARK (c1=64, c2=128)")
    print("=" * 80)
    # c1 = 64, c2 = 128
    model = cdll.deep_dhvn_create(128, 128, 64, 128, 128)
    cdll.deep_dhvn_set_substeps(model, 2)
    mem_kb = cdll.deep_dhvn_get_memory_footprint(model) / 1024.0
    print(f"Total Working Memory: {mem_kb:.1f} KB (Invariant to sequence length)")

    dt = ctypes.c_float(0.02)
    lr = 0.035
    epochs = 100

    print(f"{'Epoch':<8} | {'Loss':<12} | {'Perplexity':<12} | {'Time (ms)':<10}")
    print("-" * 55)

    for ep in range(1, epochs + 1):
        t0 = time.perf_counter()
        total_loss = 0.0
        curr_lr = lr * (0.97 ** (ep // 15))

        for seq in curriculum:
            in_bytes = seq[:-1]
            tg_bytes = seq[1:]
            L = len(in_bytes)
            in_arr = (ctypes.c_uint8 * L)(*in_bytes)
            tg_arr = (ctypes.c_uint8 * L)(*tg_bytes)
            loss = cdll.deep_dhvn_train_sequence(model, in_arr, tg_arr, L, ctypes.c_float(curr_lr), dt)
            total_loss += loss

        t1 = time.perf_counter()
        mean_loss = total_loss / len(curriculum)
        step_ms = (t1 - t0) * 1000.0

        if ep % 10 == 0 or ep == 1:
            print(f"{ep:<8} | {mean_loss:<12.4f} | {math.exp(min(12.0, mean_loss)):<12.2f} | {step_ms:<10.1f}")

    print("=" * 80)
    print("Testing multi-character generation:")
    dist_buf = (ctypes.c_float * 128)()
    for p in ["The little bird ", "Water is a ", "A clever fox "]:
        p_bytes = p.encode('ascii')
        gen = list(p_bytes)
        curr = p_bytes[-1]
        for _ in range(25):
            cdll.deep_dhvn_predict_distribution(model, curr, dt, dist_buf)
            probs = [(i, dist_buf[i]) for i in range(128)]
            # greedy next token
            probs.sort(key=lambda x: x[1], reverse=True)
            next_b = probs[0][0]
            gen.append(next_b)
            curr = next_b
        res = bytes(gen).decode('ascii', errors='replace')
        print(f"  Prompt '{p}' ──► '{res}'")

    cdll.deep_dhvn_free(model)

if __name__ == "__main__":
    run_scaled_trial()

