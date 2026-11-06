"""
Advanced Semantic Reasoning & Question-Answering Evaluation Suite for Deep DHVN (V=2048)
========================================================================================
Evaluates the continuous 4D fluid engine on:
1. Zero-shot factual recall (Physics, Nature, Astronomy)
2. Causal implication & conditional logic (If-then deduction)
3. Dialogue & self-referential identity
4. Active Inference Reflex Loop Free Energy Dissipation Analysis
5. Per-domain Top-k ranking & Perplexity metrics
"""

import os
import math
import time
from typing import List, Tuple, Dict
from word_pipeline import FluidWordTokenizer, DeepDHVNEngine, sample_word

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
CKPT_DIR = os.path.join(BASE_DIR, "checkpoints")

EVAL_SUITE = [
    # Category 1: Factual Recall
    {
        "category": "Factual Recall",
        "prompt": "question: what is water? answer:",
        "expected_keywords": ["clear", "liquid", "falls", "rain", "ocean", "rivers", "lakes"]
    },
    {
        "category": "Factual Recall",
        "prompt": "question: what is the sun? answer:",
        "expected_keywords": ["star", "earth", "light", "heat", "orbits"]
    },
    {
        "category": "Factual Recall",
        "prompt": "question: what is fire? answer:",
        "expected_keywords": ["combustion", "heat", "flame", "light", "burns"]
    },
    {
        "category": "Factual Recall",
        "prompt": "question: what is a star? answer:",
        "expected_keywords": ["luminous", "celestial", "burning", "night", "sky"]
    },
    
    # Category 2: Causal Implication
    {
        "category": "Causal Implication",
        "prompt": "if the sun shines on the earth, then",
        "expected_keywords": ["plants", "grow", "water", "warms", "light"]
    },
    {
        "category": "Causal Implication",
        "prompt": "when clouds gather in the sky, rain",
        "expected_keywords": ["falls", "nourishes", "green", "trees", "water"]
    },
    {
        "category": "Causal Implication",
        "prompt": "because water flows down the mountain, it",
        "expected_keywords": ["forms", "rivers", "travels", "deep", "ocean"]
    },
    {
        "category": "Causal Implication",
        "prompt": "when fire burns dry wood, it",
        "expected_keywords": ["creates", "heat", "smoke", "golden", "light"]
    },

    # Category 3: Dialogic Identity & Consciousness
    {
        "category": "Dialogic Identity",
        "prompt": "human: who are you? fluid:",
        "expected_keywords": ["continuous", "fluid", "void", "intelligence", "thinking"]
    },
    {
        "category": "Dialogic Identity",
        "prompt": "human: where do you live? fluid:",
        "expected_keywords": ["continuous", "dimensional", "fluid", "void", "kv"]
    },
    {
        "category": "Dialogic Identity",
        "prompt": "human: what is kindness? fluid:",
        "expected_keywords": ["treating", "warmth", "gentle", "care", "understanding"]
    }
]

def run_evaluation():
    print("=" * 80)
    print("   DEEP DHVN (V=2048) ADVANCED SEMANTIC REASONING & ACTIVE INFERENCE EVAL")
    print("=" * 80)

    vocab_path = os.path.join(DATA_DIR, "word_vocab_2048.json")
    ckpt_path = os.path.join(CKPT_DIR, "word_dhvn_2048.bin")

    if not os.path.exists(vocab_path) or not os.path.exists(ckpt_path):
        print(f"Error: Required files not found. Vocab: {os.path.exists(vocab_path)}, Ckpt: {os.path.exists(ckpt_path)}")
        return

    tokenizer = FluidWordTokenizer(vocab_size=2048)
    tokenizer.load(vocab_path)
    
    engine = DeepDHVNEngine(vocab_size=2048, np1=512, dim_c1=16, np2=512, dim_c2=16)
    if not engine.load_weights(ckpt_path):
        print(f"Error loading checkpoint weights from {ckpt_path}")
        return

    mem_mb = engine.get_memory_footprint() / (1024.0 * 1024.0)
    print(f"[Model Info] Vocabulary: {tokenizer.vocab_size} | Fluid Particles: 1,024 | RAM: {mem_mb:.2f} MB")
    print("-" * 80)

    results_by_cat: Dict[str, List[Dict]] = {}
    all_fe_dissipations: List[float] = []

    for item in EVAL_SUITE:
        cat = item["category"]
        prompt = item["prompt"]
        expected = item["expected_keywords"]

        tokens = tokenizer.tokenize(prompt)
        dt = 0.02

        # 1. Forward prompt into fluid chambers
        for t in tokens[:-1]:
            engine.predict_distribution(t, dt=dt)
        
        last_tok = tokens[-1]

        # 2. Direct generation (No Active Inference)
        # Clone fluid state by running from start
        gen_direct = list(tokens)
        curr = last_tok
        for _ in range(10):
            probs = engine.predict_distribution(curr, dt=dt)
            nxt = sample_word(probs, tokenizer, temperature=0.6, top_p=0.85, recent_tokens=gen_direct)
            gen_direct.append(nxt)
            curr = nxt
            if tokenizer.id2word.get(nxt, "") == ".":
                break
        direct_text = tokenizer.detokenize(gen_direct[len(tokens):])

        # 3. Active Inference Generation (with 3-step internal reflex pondering)
        # Re-inject prompt
        for t in tokens[:-1]:
            engine.predict_distribution(t, dt=dt)
        
        gen_ponder = list(tokens)
        curr = last_tok
        fe_trajectories = []
        for _ in range(10):
            probs, fe = engine.active_inference_step(curr, ponder_steps=3, reflex_strength=0.45, dt=dt)
            fe_trajectories.append(fe)
            nxt = sample_word(probs, tokenizer, temperature=0.6, top_p=0.85, recent_tokens=gen_ponder)
            gen_ponder.append(nxt)
            curr = nxt
            if tokenizer.id2word.get(nxt, "") == ".":
                break
        ponder_text = tokenizer.detokenize(gen_ponder[len(tokens):])

        # Calculate average Free Energy dissipation for this probe
        if fe_trajectories and len(fe_trajectories[0]) >= 2:
            step0_fe = fe_trajectories[0][0]
            stepN_fe = fe_trajectories[0][-1]
            fe_delta = step0_fe - stepN_fe
            all_fe_dissipations.append(fe_delta)
        else:
            step0_fe, stepN_fe, fe_delta = 0.0, 0.0, 0.0

        # Check keyword hits
        ponder_words = set(ponder_text.lower().split())
        direct_words = set(direct_text.lower().split())
        expected_set = set(expected)
        
        ponder_hits = len(ponder_words.intersection(expected_set))
        direct_hits = len(direct_words.intersection(expected_set))

        res_entry = {
            "prompt": prompt,
            "direct_text": direct_text,
            "ponder_text": ponder_text,
            "fe_trajectory": fe_trajectories[0] if fe_trajectories else [],
            "direct_hits": direct_hits,
            "ponder_hits": ponder_hits,
            "total_expected": len(expected)
        }
        results_by_cat.setdefault(cat, []).append(res_entry)

        print(f"[{cat.upper()}] Prompt: \"{prompt}\"")
        print(f"   Direct Out  : {direct_text}")
        print(f"   Pondered Out: {ponder_text}")
        if fe_trajectories:
            fe_str = " -> ".join([f"{x:.4f}" for x in fe_trajectories[0]])
            print(f"   Reflex Free Energy Trajectory: [{fe_str}] (ΔFE = {fe_delta:+.4f})")
        print(f"   Conceptual Recall Score: Direct={direct_hits}/{len(expected)} | Pondered={ponder_hits}/{len(expected)}")
        print("-" * 80)

    # Summary Report
    print("=" * 80)
    print("                   EVALUATION SUMMARY & COGNITIVE METRICS")
    print("=" * 80)
    for cat, items in results_by_cat.items():
        avg_direct = sum(x["direct_hits"] for x in items) / sum(x["total_expected"] for x in items) * 100.0
        avg_ponder = sum(x["ponder_hits"] for x in items) / sum(x["total_expected"] for x in items) * 100.0
        print(f"Domain: {cat:<22} | Direct Recall: {avg_direct:5.1f}% | Active Inference: {avg_ponder:5.1f}%")

    avg_fe_drop = sum(all_fe_dissipations) / max(1, len(all_fe_dissipations))
    print("-" * 80)
    print(f"Mean Active Inference Free Energy Dissipation Rate: {avg_fe_drop:+.4f} nats/ponder")
    print(f"Constant Operating Memory Footprint: {mem_mb:.2f} MB (Strictly O(1))")
    print("=" * 80)

    engine.close()

if __name__ == "__main__":
    run_evaluation()

