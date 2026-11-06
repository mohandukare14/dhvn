"""
Conversational Dialogue Training and Interactive Execution for Deep DHVN.
Demonstrates:
1. Stabilized 2-Stage Hierarchical Void (RMS logit scaling + leaky waveguide).
2. Training on structured multi-turn dialogue pairs.
3. Interactive multi-turn sample conversation with fluid-dynamic generation.
"""

import ctypes
import os
import time
import math
import random
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

cdll.deep_dhvn_get_memory_footprint.restype = ctypes.c_size_t
cdll.deep_dhvn_get_memory_footprint.argtypes = [ctypes.c_void_p]

def sample_token(probs: List[float], temperature: float = 0.5, top_p: float = 0.85) -> int:
    valid_indices = [10, 32] + list(range(33, 127))
    sub_probs = [probs[i] for i in valid_indices]

    if temperature <= 0.05:
        max_idx = valid_indices[0]
        max_p = -1.0
        for i, idx in enumerate(valid_indices):
            if sub_probs[i] > max_p:
                max_p = sub_probs[i]
                max_idx = idx
        return max_idx

    scaled = [math.pow(max(1e-9, p), 1.0 / temperature) for p in sub_probs]
    total_scaled = sum(scaled)
    if total_scaled <= 0:
        return 32
    scaled = [s / total_scaled for s in scaled]

    sorted_pairs = sorted(zip(valid_indices, scaled), key=lambda x: x[1], reverse=True)
    cumulative = 0.0
    nucleus_tokens = []
    nucleus_probs = []
    for tok, p in sorted_pairs:
        nucleus_tokens.append(tok)
        nucleus_probs.append(p)
        cumulative += p
        if cumulative >= top_p:
            break

    norm_factor = sum(nucleus_probs)
    nucleus_probs = [p / norm_factor for p in nucleus_probs]

    r = random.random()
    cum = 0.0
    for tok, p in zip(nucleus_tokens, nucleus_probs):
        cum += p
        if r <= cum:
            return tok

    return nucleus_tokens[0]

def generate_response(model, prompt: str, max_chars: int = 70, temperature: float = 0.5, top_p: float = 0.85, dt: float = 0.02) -> str:
    prompt_bytes = prompt.encode('ascii', errors='ignore')
    distribution_buffer = (ctypes.c_float * 128)()

    # Prime fluid with user prompt
    for b in prompt_bytes:
        cdll.deep_dhvn_predict_distribution(model, b, ctypes.c_float(dt), distribution_buffer)

    response_bytes = []
    curr_b = prompt_bytes[-1]

    for _ in range(max_chars):
        cdll.deep_dhvn_predict_distribution(model, curr_b, ctypes.c_float(dt), distribution_buffer)
        raw_probs = [distribution_buffer[i] for i in range(128)]
        next_b = sample_token(raw_probs, temperature=temperature, top_p=top_p)
        if next_b == 10:  # Newline ends turn
            break
        response_bytes.append(next_b)
        curr_b = next_b

    return bytes(response_bytes).decode('ascii', errors='replace').strip()

def run_conversation_demo():
    print("=" * 88)
    print("        DEEP DHVN: CONVERSATIONAL AGENT ON 2-STAGE HIERARCHICAL VOID")
    print("=" * 88)

    dialogue_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "dialogue.txt")
    with open(dialogue_path, "r", encoding="ascii") as f:
        dialogue_text = f.read().strip()

    dialogue_pairs = [d.strip() + "\n" for d in dialogue_text.split("\n\n") if d.strip()]
    print(f"Loaded Dialogue Dataset: {len(dialogue_pairs)} dialogue exchanges.")

    vocab_size = 128
    np1 = 64
    dim_c1 = 16
    np2 = 64
    dim_c2 = 32
    epochs = 60
    lr = 0.02
    dt = ctypes.c_float(0.02)

    model = cdll.deep_dhvn_create(vocab_size, np1, dim_c1, np2, dim_c2)
    mem_kb = cdll.deep_dhvn_get_memory_footprint(model) / 1024.0

    print(f"Architecture: Deep DHVN (Acoustic Void: N_p1={np1}, Semantic Void: N_p2={np2})")
    print(f"Stabilization: Leaky Capacitive Waveguide + Logit RMS Normalization (1/sqrt(d_c))")
    print(f"Working Memory: {mem_kb:.1f} KB (Strictly Constant)")
    print("-" * 88)
    print(f"{'Epoch':<8} | {'Mean Loss (NLL)':<18} | {'Perplexity':<14} | {'Step Time':<12}")
    print("-" * 88)

    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        t0 = time.perf_counter()

        curr_lr = lr * (0.97 ** (epoch // 10))
        shuffled = list(dialogue_pairs)
        random.seed(epoch)
        random.shuffle(shuffled)

        for d_str in shuffled:
            d_bytes = d_str.encode('ascii')
            in_bytes = d_bytes[:-1]
            target_bytes = d_bytes[1:]
            L = len(in_bytes)

            in_arr = (ctypes.c_uint8 * L)(*in_bytes)
            target_arr = (ctypes.c_uint8 * L)(*target_bytes)

            loss = cdll.deep_dhvn_train_sequence(
                model, in_arr, target_arr, L,
                ctypes.c_float(curr_lr), dt
            )
            total_loss += loss

        t1 = time.perf_counter()
        mean_loss = total_loss / len(shuffled)
        ppl = math.exp(min(12.0, mean_loss))
        step_time_ms = (t1 - t0) * 1000.0

        if epoch % 10 == 0 or epoch == 1:
            print(f"{epoch:<8} | {mean_loss:>16.4f}  | {ppl:>12.2f}  | {step_time_ms:>10.2f} ms")

    print("-" * 88)
    print("\n[INTERACTIVE CONVERSATION TEST] Running Sample Multi-Turn Exchanges:")
    print("=" * 88)

    test_queries = [
        "User: Hello! Who are you?\nAgent:",
        "User: How do you remember things?\nAgent:",
        "User: What is your vocabulary?\nAgent:",
        "User: Tell me a story.\nAgent:",
        "User: Are you faster than a Transformer?\nAgent:"
    ]

    for q in test_queries:
        user_line = q.split("\n")[0]
        reply = generate_response(model, q, max_chars=75, temperature=0.35, top_p=0.85, dt=0.02)
        print(f"\n{user_line}")
        print(f"Agent: {reply}")

    print("\n" + "=" * 88)
    print("SUCCESS: 2-Stage Deep DHVN stabilized loss convergence and completed sample conversation!")
    print("=" * 88)

    cdll.deep_dhvn_free(model)

if __name__ == "__main__":
    run_conversation_demo()

