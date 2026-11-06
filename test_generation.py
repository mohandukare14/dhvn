"""
Inspect generation step-by-step to diagnose early exit / token probabilities.
"""

import ctypes
import os
import math
from typing import List

LIB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "lib")
DEEP_LIB_PATH = os.path.join(LIB_DIR, "libdhvn_deep.so")
cdll = ctypes.CDLL(DEEP_LIB_PATH)

cdll.deep_dhvn_create.restype = ctypes.c_void_p
cdll.deep_dhvn_create.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int]
cdll.deep_dhvn_free.restype = None
cdll.deep_dhvn_free.argtypes = [ctypes.c_void_p]

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

def run_step_inspection():
    curriculum = [
        b"The little bird was warm and happy.\n",
        b"Water is a clear liquid that flows into the ocean.\n"
    ]
    model = cdll.deep_dhvn_create(128, 128, 32, 128, 64)
    dt = ctypes.c_float(0.02)

    # Train for 40 epochs
    for ep in range(40):
        for seq in curriculum:
            in_b = seq[:-1]
            tg_b = seq[1:]
            in_arr = (ctypes.c_uint8 * len(in_b))(*in_b)
            tg_arr = (ctypes.c_uint8 * len(tg_b))(*tg_b)
            cdll.deep_dhvn_train_sequence(model, in_arr, tg_arr, len(in_b), ctypes.c_float(0.03), dt)

    prompt = b"The little bird "
    dist_buf = (ctypes.c_float * 128)()

    for b in prompt:
        cdll.deep_dhvn_predict_distribution(model, b, dt, dist_buf)

    print("Top 5 candidates after prompt 'The little bird ':")
    probs = [(i, dist_buf[i]) for i in range(128)]
    probs.sort(key=lambda x: x[1], reverse=True)
    for i, (byte_idx, p) in enumerate(probs[:5]):
        char_repr = repr(chr(byte_idx)) if 32 <= byte_idx <= 126 else f"\\x{byte_idx:02x}"
        print(f"  {i+1}. Byte {byte_idx} ({char_repr}): P = {p*100:.2f}%")

    cdll.deep_dhvn_free(model)

if __name__ == "__main__":
    run_step_inspection()

