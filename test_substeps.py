"""
Comparison test: 1 step vs 2 substeps vs 3 substeps in Deep DHVN.
Verify physical vortex attractor stabilization.
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

cdll.deep_dhvn_train_sequence.restype = ctypes.c_float
cdll.deep_dhvn_train_sequence.argtypes = [
    ctypes.c_void_p,
    ctypes.POINTER(ctypes.c_uint8),
    ctypes.POINTER(ctypes.c_uint8),
    ctypes.c_int,
    ctypes.c_float,
    ctypes.c_float
]

curriculum = [
    b"The little bird was warm and happy.\n",
    b"Water is a clear liquid that flows into the ocean.\n",
    b"A clever fox jumped over the river bank.\n"
]

def run_substep_trial(substeps: int, epochs: int = 35):
    print(f"\n--- Running Trial with substeps = {substeps} ---")
    model = cdll.deep_dhvn_create(128, 128, 32, 128, 64)
    cdll.deep_dhvn_set_substeps(model, substeps)
    dt = ctypes.c_float(0.02)
    lr = 0.035

    t0 = time.perf_counter()
    final_loss = 0.0
    for ep in range(1, epochs + 1):
        total_loss = 0.0
        for seq in curriculum:
            in_bytes = seq[:-1]
            tg_bytes = seq[1:]
            L = len(in_bytes)
            in_arr = (ctypes.c_uint8 * L)(*in_bytes)
            tg_arr = (ctypes.c_uint8 * L)(*tg_bytes)
            loss = cdll.deep_dhvn_train_sequence(model, in_arr, tg_arr, L, ctypes.c_float(lr), dt)
            total_loss += loss
        final_loss = total_loss / len(curriculum)
        if ep % 10 == 0 or ep == epochs:
            print(f"  Epoch {ep:2d} | Loss: {final_loss:.4f} | Perplexity: {math.exp(final_loss):.2f}")

    t1 = time.perf_counter()
    print(f"Total time: {(t1 - t0):.2f}s | Final Loss: {final_loss:.4f}")
    cdll.deep_dhvn_free(model)
    return final_loss

if __name__ == "__main__":
    print("=" * 60)
    print("STEP 1: SPH SUBSTEP RELAXATION BENCHMARK")
    print("=" * 60)
    loss1 = run_substep_trial(1, epochs=30)
    loss2 = run_substep_trial(2, epochs=30)
    loss3 = run_substep_trial(3, epochs=30)
    print("\nSummary:")
    print(f"  1 Step:   Loss = {loss1:.4f}")
    print(f"  2 Steps:  Loss = {loss2:.4f}")
    print(f"  3 Steps:  Loss = {loss3:.4f}")

