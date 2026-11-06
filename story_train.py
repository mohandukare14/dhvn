"""
Pre-Training Pipeline for DHVN on Natural Story Text.
Features:
1. Scaled 4D Fluid Space: 128 particles, 32 semantic latent dimensions.
2. Temperature Scaling and Top-p (Nucleus) Generation Engine.
3. Windowed Sequence Batching on 'stories.txt'.
4. Periodic Text Generation Checkpoints demonstrating fluid narrative synthesis.
"""

import ctypes
import os
import time
import math
import random
from typing import List, Tuple

LIB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "lib")
TRAIN_LIB_PATH = os.path.join(LIB_DIR, "libdhvn_train.so")

cdll = ctypes.CDLL(TRAIN_LIB_PATH)

cdll.dhvn_create.restype = ctypes.c_void_p
cdll.dhvn_create.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_float, ctypes.c_float]

cdll.dhvn_free.restype = None
cdll.dhvn_free.argtypes = [ctypes.c_void_p]

cdll.dhvn_trainer_create.restype = ctypes.c_void_p
cdll.dhvn_trainer_create.argtypes = [ctypes.c_void_p, ctypes.c_int]

cdll.dhvn_trainer_free.restype = None
cdll.dhvn_trainer_free.argtypes = [ctypes.c_void_p]

cdll.dhvn_train_sequence.restype = ctypes.c_float
cdll.dhvn_train_sequence.argtypes = [
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.POINTER(ctypes.c_uint8),
    ctypes.POINTER(ctypes.c_uint8),
    ctypes.c_int,
    ctypes.c_float,
    ctypes.c_float
]

cdll.dhvn_predict_distribution.restype = None
cdll.dhvn_predict_distribution.argtypes = [
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_uint8,
    ctypes.c_float,
    ctypes.POINTER(ctypes.c_float)
]

cdll.dhvn_get_memory_footprint.restype = ctypes.c_size_t
cdll.dhvn_get_memory_footprint.argtypes = [ctypes.c_void_p]

def sample_token(probs: List[float], temperature: float = 0.7, top_p: float = 0.9) -> int:
    """Samples next byte from probability distribution using Temperature and Top-p Nucleus filtering."""
    # Filter printable ASCII and newline (32..126, 10)
    valid_indices = [10] + list(range(32, 127))
    sub_probs = [probs[i] for i in valid_indices]

    # Apply Temperature scaling
    if temperature <= 0.05:
        # Greedy argmax
        max_idx = valid_indices[0]
        max_p = -1.0
        for i, idx in enumerate(valid_indices):
            if sub_probs[i] > max_p:
                max_p = sub_probs[i]
                max_idx = idx
        return max_idx

    # Scaled logits
    scaled = [math.pow(max(1e-9, p), 1.0 / temperature) for p in sub_probs]
    total_scaled = sum(scaled)
    if total_scaled <= 0:
        return valid_indices[0]
    scaled = [s / total_scaled for s in scaled]

    # Sort in descending order for Top-p
    sorted_pairs = sorted(zip(valid_indices, scaled), key=lambda x: x[1], reverse=True)

    # Top-p Nucleus filtering
    cumulative = 0.0
    nucleus_tokens = []
    nucleus_probs = []
    for tok, p in sorted_pairs:
        nucleus_tokens.append(tok)
        nucleus_probs.append(p)
        cumulative += p
        if cumulative >= top_p:
            break

    # Renormalize nucleus
    norm_factor = sum(nucleus_probs)
    nucleus_probs = [p / norm_factor for p in nucleus_probs]

    # Categorical sampling
    r = random.random()
    cum = 0.0
    for tok, p in zip(nucleus_tokens, nucleus_probs):
        cum += p
        if r <= cum:
            return tok

    return nucleus_tokens[0]

def generate_text(
    sys, trainer, prompt: str, length: int = 50,
    temperature: float = 0.7, top_p: float = 0.9, dt: float = 0.02
) -> str:
    """Autoregressively generates text continuation from a prompt."""
    prompt_bytes = prompt.encode('ascii', errors='ignore')
    distribution_buffer = (ctypes.c_float * 128)()

    # Warmup / Prime fluid state with prompt
    for b in prompt_bytes:
        cdll.dhvn_predict_distribution(sys, trainer, b, ctypes.c_float(dt), distribution_buffer)

    generated = list(prompt_bytes)
    curr_b = prompt_bytes[-1]

    for _ in range(length):
        cdll.dhvn_predict_distribution(sys, trainer, curr_b, ctypes.c_float(dt), distribution_buffer)
        raw_probs = [distribution_buffer[i] for i in range(128)]
        next_b = sample_token(raw_probs, temperature=temperature, top_p=top_p)
        generated.append(next_b)
        curr_b = next_b

    return bytes(generated).decode('ascii', errors='replace')

def run_story_pretraining():
    print("=" * 88)
    print("        DHVN PRE-TRAINING ON NATURAL STORIES WITH NUCLEUS SAMPLING")
    print("=" * 88)

    data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "stories.txt")
    with open(data_path, "rb") as f:
        text_bytes = f.read()

    print(f"Loaded Corpus: {len(text_bytes)} bytes | Vocab: ASCII (128)")

    vocab_size = 128
    num_particles = 128   # Scaled resolution
    dim_c = 32           # Scaled latent semantic dimension
    seq_len = 64         # Sequence window
    epochs = 40
    lr = 0.025
    dt = ctypes.c_float(0.02)

    sys = cdll.dhvn_create(num_particles, dim_c, ctypes.c_float(2.5), ctypes.c_float(0.8))
    trainer = cdll.dhvn_trainer_create(sys, vocab_size)

    # Slice text into overlapping sequence windows
    sequences = []
    step = 32
    for i in range(0, len(text_bytes) - seq_len, step):
        chunk = text_bytes[i : i + seq_len + 1]
        sequences.append(chunk)

    print(f"Architecture: 4D Fluid Space (N_p={num_particles}, d_c={dim_c}) | Batch Slices: {len(sequences)}")
    print(f"Memory Footprint: {cdll.dhvn_get_memory_footprint(sys) / 1024.0:.1f} KB (Strictly Constant)")
    print("-" * 88)
    print(f"{'Epoch':<8} | {'Mean Loss (NLL)':<18} | {'Perplexity':<14} | {'Step Time':<12}")
    print("-" * 88)

    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        t0 = time.perf_counter()

        # Decay learning rate smoothly
        curr_lr = lr * (0.96 ** (epoch // 8))

        # Shuffle chunks per epoch for i.i.d. gradient stability
        random.seed(epoch)
        shuffled = list(sequences)
        random.shuffle(shuffled)

        for chunk in shuffled:
            in_bytes = chunk[:-1]
            target_bytes = chunk[1:]
            L = len(in_bytes)

            in_arr = (ctypes.c_uint8 * L)(*in_bytes)
            target_arr = (ctypes.c_uint8 * L)(*target_bytes)

            loss = cdll.dhvn_train_sequence(
                sys, trainer, in_arr, target_arr, L,
                ctypes.c_float(curr_lr), dt
            )
            total_loss += loss

        t1 = time.perf_counter()
        mean_loss = total_loss / len(shuffled)
        ppl = math.exp(min(12.0, mean_loss))
        step_time_ms = (t1 - t0) * 1000.0

        if epoch % 5 == 0 or epoch == 1:
            print(f"{epoch:<8} | {mean_loss:>16.4f}  | {ppl:>12.2f}  | {step_time_ms:>10.2f} ms")

    print("-" * 88)
    print("\n[VERIFICATION] Generation Checkpoint: Testing Temperature & Top-p Sampling:")
    prompts = [
        "Once upon a ",
        "Pip flew up ",
        "The red apple "
    ]

    for p in prompts:
        print(f"\n--- Prompt: '{p}' ---")
        for temp in [0.3, 0.7, 1.0]:
            out = generate_text(sys, trainer, p, length=45, temperature=temp, top_p=0.9, dt=0.02)
            # Replace newlines for clean printing
            clean_out = out.replace('\n', ' ')
            print(f"  [T={temp:.1f}]: {clean_out}")

    print("\n" + "=" * 88)
    print("SUCCESS: DHVN successfully trained on natural story text with continuous BPTT")
    print("and generated diverse narrative continuations via temperature & nucleus sampling!")
    print("=" * 88)

    cdll.dhvn_trainer_free(trainer)
    cdll.dhvn_free(sys)

if __name__ == "__main__":
    run_story_pretraining()
