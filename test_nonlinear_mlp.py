"""
Test Non-Linear Resonant MLP Readout on Deep DHVN.
Verify if it breaks the 2.97 loss barrier.
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

def run_test():
    print("=" * 75)
    print("STEP 2: NON-LINEAR RESONANT MLP READOUT BENCHMARK")
    print("=" * 75)
    model = cdll.deep_dhvn_create(128, 128, 32, 128, 64)
    cdll.deep_dhvn_set_substeps(model, 2)
    mem_kb = cdll.deep_dhvn_get_memory_footprint(model) / 1024.0
    print(f"Total Working Memory: {mem_kb:.1f} KB (including 2-Layer MLP Readout)")

    dt = ctypes.c_float(0.02)
    lr = 0.030
    epochs = 60

    print(f"{'Epoch':<8} | {'Loss':<12} | {'Perplexity':<12} | {'Time (ms)':<10}")
    print("-" * 50)

    for ep in range(1, epochs + 1):
        t0 = time.perf_counter()
        total_loss = 0.0
        curr_lr = lr * (0.98 ** (ep // 10))

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

        if ep % 5 == 0 or ep == 1:
            print(f"{ep:<8} | {mean_loss:<12.4f} | {math.exp(min(12.0, mean_loss)):<12.2f} | {step_ms:<10.1f}")

    print("=" * 75)
    print("Generating from prompts:")
    dist_buf = (ctypes.c_float * 128)()
    for p in ["The little bird ", "Water is a ", "A clever fox "]:
        p_bytes = p.encode('ascii')
        for b in p_bytes:
            cdll.deep_dhvn_predict_distribution(model, b, dt, dist_buf)
        probs = [(i, dist_buf[i]) for i in range(128)]
        probs.sort(key=lambda x: x[1], reverse=True)
        top3 = [f"'{chr(idx)}'({prob*100:.1f}%)" if 32 <= idx <= 126 else f"{idx}" for idx, prob in probs[:4]]
        print(f"  Prompt '{p}' ──► Top candidates: {', '.join(top3)}")

    cdll.deep_dhvn_free(model)

if __name__ == "__main__":
    run_test()

