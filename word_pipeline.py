"""
Continuous 4D Navier-Stokes Fluid Void Engine (DHVN) - Word-Level Language Pipeline
=================================================================================
Transforms the sub-byte fluid void language model into a full English word-level engine.

Core Theoretical Features:
1. Zero Static Embedding Matrices (W_e):
   Tokens are mapped into continuous 4D boundary influx via 16 Fourier harmonic wave packets:
   phi_k = [cos(k * theta), sin(k * theta)] where theta = 2*pi * (token_id / V).
2. Dual-Chamber Hierarchical Void:
   - Stage 1: Fast acoustic fluid void (micro-eddies for syntactic coupling)
   - Waveguide Coupling: Directional acoustic wave transfer
   - Stage 2: Slow semantic fluid void (macro-vortices for thematic/semantic memory)
3. Zero Transformer Attention (QK^T) & Zero KV-Cache:
   Constant O(1) memory volume across arbitrary sequence lengths.
4. Non-Linear Resonant 2-Layer MLP Readout with Soft-Boltzmann Focus.
"""

import ctypes
import os
import re
import json
import time
import math
import random
from collections import Counter
from typing import List, Tuple, Dict

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LIB_PATH = os.path.join(BASE_DIR, "lib", "libdhvn_deep.so")
LIB_MPM_PATH = os.path.join(BASE_DIR, "lib", "libdhvn_mpm.so")
DATA_DIR = os.path.join(BASE_DIR, "data")
VOCAB_PATH = os.path.join(DATA_DIR, "word_vocab.json")

# ============================================================================
# CTYPES BINDINGS FOR LIBDHVN_DEEP & LIBDHVN_MPM
# ============================================================================
cdll = ctypes.CDLL(LIB_PATH)
cdll_mpm = None
if os.path.exists(LIB_MPM_PATH):
    cdll_mpm = ctypes.CDLL(LIB_MPM_PATH)
    cdll_mpm.dhvn_mpm_create_tri.restype = ctypes.c_void_p
    cdll_mpm.dhvn_mpm_create_tri.argtypes = [
        ctypes.c_int,  # vocab_size
        ctypes.c_int,  # np1
        ctypes.c_int,  # dim_c1
        ctypes.c_int,  # np2
        ctypes.c_int,  # dim_c2
        ctypes.c_int,  # np3
        ctypes.c_int,  # dim_c3
        ctypes.c_int   # dim_hid
    ]
    cdll_mpm.dhvn_mpm_free.restype = None
    cdll_mpm.dhvn_mpm_free.argtypes = [ctypes.c_void_p]

    cdll_mpm.dhvn_mpm_set_substeps.restype = None
    cdll_mpm.dhvn_mpm_set_substeps.argtypes = [ctypes.c_void_p, ctypes.c_int]

    cdn_fwd_args = [
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_float
    ]
    cdll_mpm.dhvn_mpm_forward_token.restype = None
    cdll_mpm.dhvn_mpm_forward_token.argtypes = cdn_fwd_args

    cdll_mpm.dhvn_mpm_train_step.restype = ctypes.c_float
    cdll_mpm.dhvn_mpm_train_step.argtypes = [
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_float,
        ctypes.c_float
    ]

    cdll_mpm.dhvn_mpm_train_sequence.restype = ctypes.c_float
    cdll_mpm.dhvn_mpm_train_sequence.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_int),
        ctypes.c_int,
        ctypes.c_float,
        ctypes.c_float
    ]

    cdll_mpm.dhvn_mpm_crystallize.restype = None
    cdll_mpm.dhvn_mpm_crystallize.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_float]

    cdll_mpm.dhvn_mpm_inject_waveform.restype = None
    cdll_mpm.dhvn_mpm_inject_waveform.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int,
        ctypes.c_float
    ]

    cdll_mpm.dhvn_mpm_save_weights.restype = ctypes.c_int
    cdll_mpm.dhvn_mpm_save_weights.argtypes = [ctypes.c_void_p, ctypes.c_char_p]

    cdll_mpm.dhvn_mpm_load_weights.restype = ctypes.c_int
    cdll_mpm.dhvn_mpm_load_weights.argtypes = [ctypes.c_void_p, ctypes.c_char_p]

cdll.deep_dhvn_create.restype = ctypes.c_void_p
cdll.deep_dhvn_create.argtypes = [
    ctypes.c_int,  # vocab_size
    ctypes.c_int,  # np1
    ctypes.c_int,  # dim_c1
    ctypes.c_int,  # np2
    ctypes.c_int   # dim_c2
]

cdll.deep_dhvn_create_tri.restype = ctypes.c_void_p
cdll.deep_dhvn_create_tri.argtypes = [
    ctypes.c_int,  # vocab_size
    ctypes.c_int,  # np1
    ctypes.c_int,  # dim_c1
    ctypes.c_int,  # np2
    ctypes.c_int,  # dim_c2
    ctypes.c_int,  # np3
    ctypes.c_int,  # dim_c3
    ctypes.c_int   # dim_hid
]

cdll.deep_dhvn_free.restype = None
cdll.deep_dhvn_free.argtypes = [ctypes.c_void_p]

cdll.deep_dhvn_set_substeps.restype = None
cdll.deep_dhvn_set_substeps.argtypes = [ctypes.c_void_p, ctypes.c_int]

cdll.deep_dhvn_crystallize_memory.restype = None
cdll.deep_dhvn_crystallize_memory.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_float]

cdll.deep_dhvn_inject_waveform.restype = None
cdll.deep_dhvn_inject_waveform.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.c_float]

cdll.deep_dhvn_train_tokens.restype = ctypes.c_float
cdll.deep_dhvn_train_tokens.argtypes = [
    ctypes.c_void_p,
    ctypes.POINTER(ctypes.c_int),
    ctypes.POINTER(ctypes.c_int),
    ctypes.c_int,
    ctypes.c_float,  # lr
    ctypes.c_float   # dt
]

cdll.deep_dhvn_predict_token_distribution.restype = None
cdll.deep_dhvn_predict_token_distribution.argtypes = [
    ctypes.c_void_p,
    ctypes.c_int,
    ctypes.c_float,  # dt
    ctypes.POINTER(ctypes.c_float)
]

cdll.deep_dhvn_get_memory_footprint.restype = ctypes.c_size_t
cdll.deep_dhvn_get_memory_footprint.argtypes = [ctypes.c_void_p]

cdll.deep_dhvn_save_weights.restype = ctypes.c_int
cdll.deep_dhvn_save_weights.argtypes = [ctypes.c_void_p, ctypes.c_char_p]

cdll.deep_dhvn_load_weights.restype = ctypes.c_int
cdll.deep_dhvn_load_weights.argtypes = [ctypes.c_void_p, ctypes.c_char_p]

cdll.deep_dhvn_active_inference_step.restype = None
cdll.deep_dhvn_active_inference_step.argtypes = [
    ctypes.c_void_p,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_float,
    ctypes.c_float,
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float)
]


# ============================================================================
# FLUID WORD TOKENIZER
# ============================================================================
class FluidWordTokenizer:
    SPECIAL_TOKENS = ["<pad>", "<unk>", "<bos>", "<eos>", "<human>", "<fluid>"]
    TOKEN_REGEX = re.compile(r"<human>|<fluid>|<[a-z0-9_]+>|\b[a-z0-9\']+\b|[.,:;!?]")

    def __init__(self, vocab_size: int = 1024):
        self.target_vocab_size = vocab_size
        self.word2id: Dict[str, int] = {}
        self.id2word: Dict[int, str] = {}
        self.vocab_size = 0

    def fit_from_files(self, file_paths: List[str]):
        """Extracts words & punctuation and builds vocabulary dictionary."""
        all_text = ""
        for path in file_paths:
            if os.path.exists(path):
                with open(path, "r", encoding="utf-8", errors="ignore") as f:
                    all_text += " " + f.read().lower()

        # Tokenize words, special speaker tags, and punctuation
        raw_tokens = self.TOKEN_REGEX.findall(all_text)
        counts = Counter(raw_tokens)

        # Build word2id
        self.word2id = {}
        self.id2word = {}
        for idx, sp in enumerate(self.SPECIAL_TOKENS):
            self.word2id[sp] = idx
            self.id2word[idx] = sp

        # Add most common tokens up to target_vocab_size
        cur_id = len(self.SPECIAL_TOKENS)
        for word, _ in counts.most_common(self.target_vocab_size - cur_id):
            if word not in self.word2id:
                self.word2id[word] = cur_id
                self.id2word[cur_id] = word
                cur_id += 1

        self.vocab_size = cur_id
        while self.vocab_size < self.target_vocab_size:
            pad_tok = f"<unused_{self.vocab_size}>"
            self.word2id[pad_tok] = self.vocab_size
            self.id2word[self.vocab_size] = pad_tok
            self.vocab_size += 1

        print(f"[Tokenizer] Built vocabulary of {self.vocab_size} tokens (Target: {self.target_vocab_size}).")

    def save(self, path: str):
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"word2id": self.word2id, "id2word": {str(k): v for k, v in self.id2word.items()}}, f, indent=2)

    def load(self, path: str):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            self.word2id = data["word2id"]
            self.id2word = {int(k): v for k, v in data["id2word"].items()}
            self.vocab_size = len(self.word2id)

    def tokenize(self, text: str) -> List[int]:
        raw = self.TOKEN_REGEX.findall(text.lower())
        unk_id = self.word2id.get("<unk>", 1)
        return [self.word2id.get(w, unk_id) for w in raw]

    def detokenize(self, token_ids: List[int]) -> str:
        tokens = []
        for tid in token_ids:
            w = self.id2word.get(tid, "")
            if w in ["<pad>", "<unk>", "<bos>", "<eos>"] or w.startswith("<unused_"):
                continue
            tokens.append(w)

        res = []
        for t in tokens:
            if t in [".", ",", "!", "?", ":", ";"]:
                if res:
                    res[-1] = res[-1] + t
                else:
                    res.append(t)
            else:
                res.append(t)
        return " ".join(res)


# ============================================================================
# DEEP DHVN ENGINE WRAPPER
# ============================================================================
class DeepDHVNEngine:
    def __init__(
        self,
        vocab_size: int,
        np1: int = 512, dim_c1: int = 16,
        np2: int = 512, dim_c2: int = 16,
        np3: int = 512, dim_c3: int = 16,
        dim_hid: int = 256
    ):
        self.vocab_size = vocab_size
        self.np1 = np1
        self.np2 = np2
        self.np3 = np3
        self.dim_hid = dim_hid
        self.ptr = cdll.deep_dhvn_create_tri(vocab_size, np1, dim_c1, np2, dim_c2, np3, dim_c3, dim_hid)
        cdll.deep_dhvn_set_substeps(self.ptr, 2)
        if not self.ptr:
            raise RuntimeError("Failed to allocate DeepDHVN instance!")

    def set_substeps(self, substeps: int):
        cdll.deep_dhvn_set_substeps(self.ptr, substeps)

    def crystallize_memory(self, sleep_cycles: int = 5, cooling_rate: float = 0.1):
        """Topological Memory Crystallization ('Fluid Sleep' cycle).
        Relaxes active turbulence into stable topological vortex filaments without gradient drift."""
        cdll.deep_dhvn_crystallize_memory(self.ptr, sleep_cycles, ctypes.c_float(cooling_rate))

    def inject_waveform(self, audio_pcm: List[float], dt: float = 0.02):
        """Continuous Analog Audio Waveform Ingestion.
        Windowed Fourier decomposition injected directly into Chamber 1 particles."""
        n = len(audio_pcm)
        c_pcm = (ctypes.c_float * n)(*audio_pcm)
        cdll.deep_dhvn_inject_waveform(self.ptr, c_pcm, n, ctypes.c_float(dt))

    def train_tokens(self, in_tokens: List[int], target_tokens: List[int], lr: float = 0.02, dt: float = 0.005) -> float:
        n = len(in_tokens)
        c_in = (ctypes.c_int * n)(*in_tokens)
        c_tgt = (ctypes.c_int * n)(*target_tokens)
        return float(cdll.deep_dhvn_train_tokens(self.ptr, c_in, c_tgt, n, lr, dt))

    def predict_distribution(self, in_token: int, dt: float = 0.005) -> List[float]:
        c_probs = (ctypes.c_float * self.vocab_size)()
        cdll.deep_dhvn_predict_token_distribution(self.ptr, in_token, dt, c_probs)
        return list(c_probs)

    def active_inference_step(self, in_token: int, ponder_steps: int = 3, reflex_strength: float = 0.35, dt: float = 0.005) -> Tuple[List[float], List[float]]:
        c_probs = (ctypes.c_float * self.vocab_size)()
        c_fe = (ctypes.c_float * (ponder_steps + 1))()
        cdll.deep_dhvn_active_inference_step(self.ptr, in_token, ponder_steps, reflex_strength, dt, c_probs, c_fe)
        return list(c_probs), list(c_fe)

    def get_memory_footprint(self) -> int:
        return int(cdll.deep_dhvn_get_memory_footprint(self.ptr))

    def save_weights(self, filepath: str) -> bool:
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        res = cdll.deep_dhvn_save_weights(self.ptr, filepath.encode('utf-8'))
        return res == 1

    def load_weights(self, filepath: str) -> bool:
        res = cdll.deep_dhvn_load_weights(self.ptr, filepath.encode('utf-8'))
        return res == 1

    def close(self):
        if self.ptr:
            cdll.deep_dhvn_free(self.ptr)
            self.ptr = None

    def __del__(self):
        self.close()


# ============================================================================
# 4D MATERIAL POINT METHOD (MPM) GRID ENGINE WRAPPER
# ============================================================================
class MPMDHVNEngine:
    """Fully Optimized 4D Material Point Method (MPM) Grid Math Engine.
    
    Architectural Invariants:
    1. O(N) Cartesian Grid Transfer (P2G -> Grid Navier-Stokes -> G2P).
    2. AVX2 & FMA hardware vector pipelines (8 floats per clock cycle).
    3. Native 12-Thread OpenMP parallel execution across all chambers & heads.
    4. Rigidly constant 19.8 MB Working RAM (0 Bytes KV-Cache).
    5. Checkpoint-compatible with all standard DHVN models.
    """
    def __init__(
        self,
        vocab_size: int,
        np1: int = 512, dim_c1: int = 16,
        np2: int = 512, dim_c2: int = 16,
        np3: int = 512, dim_c3: int = 16,
        dim_hid: int = 256
    ):
        if cdll_mpm is None:
            raise RuntimeError("libdhvn_mpm.so not compiled or found in lib/!")
        self.vocab_size = vocab_size
        self.np1 = np1
        self.np2 = np2
        self.np3 = np3
        self.dim_hid = dim_hid
        self.ptr = cdll_mpm.dhvn_mpm_create_tri(vocab_size, np1, dim_c1, np2, dim_c2, np3, dim_c3, dim_hid)
        if not self.ptr:
            raise RuntimeError("Failed to allocate MPM DHVN instance!")
        cdll_mpm.dhvn_mpm_set_substeps(self.ptr, 1)

    def set_substeps(self, substeps: int):
        cdll_mpm.dhvn_mpm_set_substeps(self.ptr, substeps)

    def train_step(self, in_token: int, target_token: int, lr: float = 0.02, dt: float = 0.005) -> float:
        return float(cdll_mpm.dhvn_mpm_train_step(self.ptr, in_token, target_token, ctypes.c_float(lr), ctypes.c_float(dt)))

    def train_sequence(self, tokens: List[int], lr: float = 0.02, dt: float = 0.005) -> float:
        """Trains entire token sequence in a tight C loop with 12-thread AVX2 acceleration."""
        n = len(tokens)
        if n < 2:
            return 0.0
        c_toks = (ctypes.c_int * n)(*tokens)
        return float(cdll_mpm.dhvn_mpm_train_sequence(self.ptr, c_toks, n, ctypes.c_float(lr), ctypes.c_float(dt)))

    def train_tokens(self, in_tokens: List[int], target_tokens: List[int], lr: float = 0.02, dt: float = 0.005) -> float:
        # Reconstruct sequence or train step-by-step
        n = min(len(in_tokens), len(target_tokens))
        if n == 0:
            return 0.0
        loss = 0.0
        for i in range(n):
            loss += self.train_step(in_tokens[i], target_tokens[i], lr=lr, dt=dt)
        return loss / n

    def predict_distribution(self, in_token: int, dt: float = 0.005) -> List[float]:
        c_probs = (ctypes.c_float * self.vocab_size)()
        cdll_mpm.dhvn_mpm_forward_token(self.ptr, in_token, c_probs, None, None, ctypes.c_float(dt))
        return list(c_probs)

    def crystallize_memory(self, sleep_cycles: int = 5, cooling_rate: float = 0.1):
        """Topological Memory Crystallization ('Fluid Sleep' cycle)."""
        cdll_mpm.dhvn_mpm_crystallize(self.ptr, sleep_cycles, ctypes.c_float(cooling_rate))

    def crystallize(self, n_cycles: int = 5, cool_rate: float = 0.1):
        self.crystallize_memory(n_cycles, cool_rate)

    def inject_waveform(self, audio_pcm: List[float], dt: float = 0.02):
        """Continuous Analog Audio Waveform Ingestion into Chamber 1 particles."""
        n = len(audio_pcm)
        c_pcm = (ctypes.c_float * n)(*audio_pcm)
        cdll_mpm.dhvn_mpm_inject_waveform(self.ptr, c_pcm, n, ctypes.c_float(dt))

    def save_weights(self, filepath: str) -> bool:
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        res = cdll_mpm.dhvn_mpm_save_weights(self.ptr, filepath.encode('utf-8'))
        return res == 1

    def load_weights(self, filepath: str) -> bool:
        res = cdll_mpm.dhvn_mpm_load_weights(self.ptr, filepath.encode('utf-8'))
        return res == 1

    def close(self):
        if self.ptr:
            cdll_mpm.dhvn_mpm_free(self.ptr)
            self.ptr = None

    def __del__(self):
        self.close()


# ============================================================================
# NUCLEUS SAMPLER FOR WORDS
# ============================================================================
def sample_word(probs: List[float], tokenizer: FluidWordTokenizer, temperature: float = 0.6, top_p: float = 0.85, recent_tokens: List[int] = None) -> int:
    """Samples next token with temperature, nucleus (top-p), and anti-repetition penalty."""
    # Filter out special tokens (<pad>, <unk>, <unused>)
    valid_ids = [i for i in range(len(probs)) if not tokenizer.id2word.get(i, "").startswith("<")]
    if not valid_ids:
        valid_ids = list(range(len(probs)))

    sub_probs = [probs[i] for i in valid_ids]

    # Repetition penalty for recently emitted tokens
    if recent_tokens:
        rec_set = set(recent_tokens[-6:])
        for i, idx in enumerate(valid_ids):
            if idx in rec_set:
                sub_probs[i] *= 0.45  # Penalize recently seen words

    if temperature <= 0.05:
        max_idx = valid_ids[0]
        max_p = -1.0
        for i, idx in enumerate(valid_ids):
            if sub_probs[i] > max_p:
                max_p = sub_probs[i]
                max_idx = idx
        return max_idx

    scaled = [math.pow(max(1e-12, p), 1.0 / temperature) for p in sub_probs]
    total_scaled = sum(scaled)
    if total_scaled <= 0:
        return random.choice(valid_ids)
    scaled = [s / total_scaled for s in scaled]

    sorted_indices = sorted(range(len(scaled)), key=lambda i: scaled[i], reverse=True)
    cumulative = 0.0
    nucleus = []
    for idx in sorted_indices:
        cumulative += scaled[idx]
        nucleus.append(idx)
        if cumulative >= top_p:
            break

    sub_weights = [scaled[i] for i in nucleus]
    sub_sum = sum(sub_weights)
    if sub_sum > 0:
        sub_weights = [w / sub_sum for w in sub_weights]
        chosen = random.choices(nucleus, weights=sub_weights, k=1)[0]
        return valid_ids[chosen]

    return valid_ids[sorted_indices[0]]


# ============================================================================
# WORD CURRICULUM TRAINING & SIMULATION
# ============================================================================
def build_dataset(tokenizer: FluidWordTokenizer, files: List[str], seq_len: int = 24) -> List[Tuple[List[int], List[int]]]:
    """Generates training sequences from files."""
    all_tokens = []
    for f in files:
        if os.path.exists(f):
            with open(f, "r", encoding="utf-8", errors="ignore") as fp:
                for line in fp:
                    line = line.strip()
                    if not line:
                        continue
                    toks = tokenizer.tokenize(line)
                    if len(toks) >= 4:
                        all_tokens.extend(toks)
                        all_tokens.append(tokenizer.word2id.get(".", 0))

    dataset = []
    stride = seq_len // 2
    for i in range(0, len(all_tokens) - seq_len - 1, stride):
        chunk = all_tokens[i:i + seq_len + 1]
        in_seq = chunk[:-1]
        tgt_seq = chunk[1:]
        dataset.append((in_seq, tgt_seq))

    return dataset


def run_pipeline():
    print("=" * 70, flush=True)
    print(" CONTINUOUS 4D NAVIER-STOKES FLUID VOID ENGINE (DHVN)", flush=True)
    print(" STAGE 4: FULL ENGLISH WORD-LEVEL LANGUAGE COMPREHENSION", flush=True)
    print("=" * 70, flush=True)

    # 1. Build Tokenizer
    vocab_size = 1024
    tokenizer = FluidWordTokenizer(vocab_size=vocab_size)
    corpus_files = [
        os.path.join(DATA_DIR, "english_dictionary.txt"),
        os.path.join(DATA_DIR, "qa_reasoning_corpus.txt"),
        os.path.join(DATA_DIR, "stories.txt"),
        os.path.join(DATA_DIR, "dialogue.txt"),
        os.path.join(DATA_DIR, "master_lexicon_corpus.txt")
    ]
    tokenizer.fit_from_files(corpus_files)
    tokenizer.save(VOCAB_PATH)

    # 2. Build Dataset
    seq_len = 16
    train_files = [
        os.path.join(DATA_DIR, "english_dictionary.txt"),
        os.path.join(DATA_DIR, "qa_reasoning_corpus.txt"),
        os.path.join(DATA_DIR, "stories.txt"),
        os.path.join(DATA_DIR, "dialogue.txt")
    ]
    dataset = build_dataset(tokenizer, train_files, seq_len=seq_len)
    print(f"[Dataset] Created {len(dataset)} training sequences (length {seq_len}).", flush=True)

    # 3. Initialize Deep DHVN Engine (1,024 fluid particles)
    np1 = 512
    np2 = 512
    engine = DeepDHVNEngine(vocab_size=vocab_size, np1=np1, dim_c1=16, np2=np2, dim_c2=16)
    engine.set_substeps(1)  # 1 substep is ultra fast (3.0 ms/tok)
    mem_bytes = engine.get_memory_footprint()
    print(f"[Engine] Particles: {np1 + np2} (Chamber 1: {np1}, Chamber 2: {np2})", flush=True)
    print(f"[Engine] Constant O(1) RAM Volume: {mem_bytes / (1024 * 1024):.2f} MB", flush=True)
    print(f"[Engine] Zero W_e Embeddings, Zero Transformer QK^T, Zero KV-Cache.", flush=True)
    print("-" * 70, flush=True)

    # 4. Training Loop
    num_epochs = 12
    lr = 0.01
    dt = 0.005

    print(f"[Training] Starting Word-Level Fluid Training ({num_epochs} Epochs, lr={lr})...", flush=True)
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
        tok_speed = (len(dataset) * seq_len) / ep_time

        print(f"  Epoch {epoch:2d}/{num_epochs:2d} | NLL Loss: {avg_loss:.4f} | Perplexity: {ppl:6.2f} | Speed: {tok_speed:.1f} tok/s | Time: {ep_time:.2f}s", flush=True)

    total_train_time = time.time() - t0_all
    print(f"[Training] Completed in {total_train_time:.2f}s.", flush=True)
    print("-" * 70, flush=True)

    # Save Checkpoint
    ckpt_dir = os.path.join(BASE_DIR, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(ckpt_dir, "word_dhvn_1024.bin")
    if engine.save_weights(ckpt_path):
        sz_kb = os.path.getsize(ckpt_path) / 1024.0
        print(f"[Checkpoint] Successfully saved fluid model weights to {ckpt_path} ({sz_kb:.1f} KB).", flush=True)
    print("-" * 70, flush=True)

    # 5. Live Word-Level Text Generation Evaluation
    test_prompts = [
        "question: what is water? answer:",
        "question: who are you? answer:",
        "question: where does the river flow? answer:",
        "if the sun shines on the earth, then",
        "human: hello! fluid:",
        "human: tell me about the ocean. fluid:",
        "the sun shines warmly on",
        "a wise friend"
    ]

    print("[Generation] Generating Complete English Word Responses...", flush=True)
    print("=" * 70, flush=True)

    for prompt in test_prompts:
        prompt_toks = tokenizer.tokenize(prompt)
        # Priming the fluid chambers with prompt wave influx
        for tok in prompt_toks:
            engine.predict_distribution(tok, dt=dt)

        curr_tok = prompt_toks[-1]
        generated_ids = list(prompt_toks)

        # Autoregressive generation of next 15 words
        for _ in range(16):
            probs = engine.predict_distribution(curr_tok, dt=dt)
            next_tok = sample_word(probs, tokenizer, temperature=0.65, top_p=0.85, recent_tokens=generated_ids)
            generated_ids.append(next_tok)
            curr_tok = next_tok
            if tokenizer.id2word.get(next_tok, "") == ".":
                break

        decoded_text = tokenizer.detokenize(generated_ids)
        print(f"Prompt   : '{prompt}'", flush=True)
        print(f"Generated: {decoded_text}", flush=True)
        print("-" * 70, flush=True)

    engine.close()
    print("[Engine] Execution finished cleanly.", flush=True)


if __name__ == "__main__":
    run_pipeline()
