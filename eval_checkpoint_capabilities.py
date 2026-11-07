"""
Evaluate Current Stage Capabilities of DHVN from pg19_stream_dhvn_4096.bin
With Top Semantic Candidate Ranking and <unk> handling.
"""

import os
import sys
import torch
import torch.nn.functional as F
from word_pipeline import FluidWordTokenizer
from gpu_pg19_train import DHVNGPUModel

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
CKPT_DIR = os.path.join(BASE_DIR, "checkpoints")

VOCAB_PATH = os.path.join(DATA_DIR, "conversational_vocab_4096.json")
CKPT_PATH = os.path.join(CKPT_DIR, "pg19_stream_dhvn_4096.bin")

def evaluate_capabilities():
    print("=" * 88)
    print("       DHVN 4D FLUID ENGINE — CURRENT STAGE CAPABILITIES & SEMANTIC EVALUATION")
    print("=" * 88)

    tokenizer = FluidWordTokenizer(vocab_size=4096)
    tokenizer.load(VOCAB_PATH)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = DHVNGPUModel(vocab_size=4096, dim_c1=16, dim_c2=16, dim_c3=16, dim_hid=256, in_dim=32).to(device)
    
    if not model.load_binary_checkpoint(CKPT_PATH):
        print("Failed to load binary checkpoint.")
        return
    model.eval()

    test_prompts = [
        "the sun is",
        "knowledge is",
        "water is a",
        "in the beginning of the",
        "the earth moves around the",
        "truth is observed through",
        "when fire burns it creates",
        "the king and the queen walked into the"
    ]

    print(f"Checkpoint : {os.path.basename(CKPT_PATH)}")
    print(f"Hardware   : {device} | Total Parameters: 1,270,444 | Vocab: {tokenizer.vocab_size}")
    print("-" * 88)
    print(f"{'Prompt':<42} | Top Semantic Predictions (in-vocab words)")
    print("-" * 88)

    with torch.no_grad():
        for prompt in test_prompts:
            tokens = tokenizer.tokenize(prompt)
            in_t = torch.tensor(tokens, dtype=torch.long, device=device)
            logits = model(in_t)[-1].clone()
            
            # Show raw top-1 including <unk>
            raw_top_idx = torch.argmax(logits).item()
            raw_word = tokenizer.id2word.get(raw_top_idx, "?")

            # Suppress <unk> and <pad> to reveal in-vocabulary semantic ranking
            logits[0] = -1e9 # <pad>
            logits[1] = -1e9 # <unk>
            probs = F.softmax(logits, dim=-1)
            
            topk_probs, topk_ids = torch.topk(probs, 6)
            candidates = [f"{tokenizer.id2word.get(idx.item(), '?')} ({prob.item()*100:.1f}%)" for prob, idx in zip(topk_probs, topk_ids)]
            
            # Autoregressive generation (5 tokens without <unk>)
            gen_tokens = list(tokens)
            for _ in range(6):
                curr_t = torch.tensor([gen_tokens[-1]], dtype=torch.long, device=device)
                curr_l = model(curr_t)[-1].clone()
                curr_l[0] = -1e9
                curr_l[1] = -1e9
                # Penalize immediate repeats
                for past_id in gen_tokens[-2:]:
                    curr_l[past_id] -= 2.0
                nxt_tok = torch.argmax(curr_l).item()
                gen_tokens.append(nxt_tok)
            
            gen_text = " ".join([tokenizer.id2word.get(t, "") for t in gen_tokens])

            print(f"\"{prompt}\"")
            print(f"  ► In-Vocab Top 6 : {', '.join(candidates[:6])}")
            print(f"  ► Fluent Rollout : \"{gen_text}\"")
            print()

if __name__ == "__main__":
    evaluate_capabilities()
