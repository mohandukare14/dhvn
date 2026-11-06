"""
Continuous 4D Navier-Stokes Fluid Void Engine (DHVN) - Advanced Interactive Console
===================================================================================
Features:
- Dual-Chamber Hierarchical Void (Stage 1 Acoustic + Stage 2 Semantic)
- Hebbian Vorticity Strain Plasticity
- Active Inference Reflex Loop (/ponder on|off)
- Real-Time Live Experience Ingestion (/ingest <text>)
- Constant O(1) Memory Footprint (Zero KV-Cache)
"""

import os
import sys
import time
from word_pipeline import FluidWordTokenizer, DeepDHVNEngine, sample_word

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
CKPT_DIR = os.path.join(BASE_DIR, "checkpoints")

def main():
    print("=" * 75)
    print(" CONTINUOUS 4D FLUID VOID ENGINE (DHVN) - ADVANCED COGNITIVE CONSOLE")
    print("=" * 75)

    # Check for 2048 model first, fallback to 1024
    ckpt_2048 = os.path.join(CKPT_DIR, "word_dhvn_2048.bin")
    vocab_2048 = os.path.join(DATA_DIR, "word_vocab_2048.json")

    ckpt_1024 = os.path.join(CKPT_DIR, "word_dhvn_1024.bin")
    vocab_1024 = os.path.join(DATA_DIR, "word_vocab.json")

    if os.path.exists(ckpt_2048) and os.path.exists(vocab_2048):
        vocab_path = vocab_2048
        ckpt_path = ckpt_2048
        vocab_size = 2048
    elif os.path.exists(ckpt_1024) and os.path.exists(vocab_1024):
        vocab_path = vocab_1024
        ckpt_path = ckpt_1024
        vocab_size = 1024
    else:
        print("Error: No trained model checkpoint found in checkpoints/")
        return

    tokenizer = FluidWordTokenizer(vocab_size=vocab_size)
    tokenizer.load(vocab_path)
    print(f"[Tokenizer] Loaded vocabulary of {tokenizer.vocab_size} tokens from {os.path.basename(vocab_path)}.")

    engine = DeepDHVNEngine(vocab_size=vocab_size, np1=512, dim_c1=16, np2=512, dim_c2=16)
    if not engine.load_weights(ckpt_path):
        print(f"Error: Failed to load weights from {ckpt_path}")
        return

    mem_mb = engine.get_memory_footprint() / (1024.0 * 1024.0)
    print(f"[Engine] Fluid Dual-Chambers Loaded (1,024 particles).")
    print(f"[Engine] Working RAM Volume: strictly {mem_mb:.2f} MB (O(1) constant volume).")
    print(f"[Engine] Checkpoint: {os.path.basename(ckpt_path)}")
    print("=" * 75)
    print("Commands:")
    print("  /ponder [on|off] [steps]  - Toggle Active Inference internal reflex pondering")
    print("  /ingest <text>           - Stream live experience directly into fluid chambers")
    print("  /stats                   - Display fluid memory & telemetry")
    print("  exit / quit              - Leave the console")
    print("-" * 75)

    ponder_enabled = False
    ponder_steps = 3
    reflex_strength = 0.35
    dt = 0.005

    while True:
        try:
            prompt = input("\nYou > ").strip()
            if not prompt:
                continue

            if prompt.lower() in ["exit", "quit", "q"]:
                print("Exiting fluid engine.")
                break

            # Command handling
            if prompt.startswith("/ponder"):
                parts = prompt.split()
                if len(parts) >= 2:
                    if parts[1].lower() in ["on", "1", "true"]:
                        ponder_enabled = True
                        if len(parts) >= 3:
                            ponder_steps = max(1, int(parts[2]))
                        print(f"[Active Inference] Pondering enabled ({ponder_steps} steps, reflex_strength={reflex_strength}).")
                    elif parts[1].lower() in ["off", "0", "false"]:
                        ponder_enabled = False
                        print("[Active Inference] Pondering disabled (Direct readout mode).")
                else:
                    ponder_enabled = not ponder_enabled
                    state_str = "enabled" if ponder_enabled else "disabled"
                    print(f"[Active Inference] Pondering {state_str} ({ponder_steps} steps).")
                continue

            if prompt.startswith("/ingest"):
                ingest_text = prompt[7:].strip()
                if not ingest_text:
                    print("Usage: /ingest <text to stream into fluid chambers>")
                    continue
                in_toks = tokenizer.tokenize(ingest_text)
                t0_ingest = time.time()
                for tok in in_toks:
                    engine.predict_distribution(tok, dt=dt)
                if len(in_toks) >= 4:
                    engine.train_tokens(in_toks[:-1], in_toks[1:], lr=0.005, dt=dt)
                d_time = time.time() - t0_ingest
                print(f"[Ingestion] Streamed {len(in_toks)} words into fluid void in {d_time*1000:.1f} ms ({len(in_toks)/max(d_time,1e-4):.1f} tok/s).")
                continue

            if prompt.startswith("/stats"):
                print(f"[Stats] Working RAM: {mem_mb:.2f} MB | Particles: 1,024 | Vocab: {tokenizer.vocab_size} | Pondering: {ponder_enabled}")
                continue

            # Standard Generation
            prompt_toks = tokenizer.tokenize(prompt)
            if not prompt_toks:
                print("Could not parse tokens from prompt.")
                continue

            # Prime fluid chamber with prompt wave influx
            t0 = time.time()
            for tok in prompt_toks:
                engine.predict_distribution(tok, dt=dt)

            curr_tok = prompt_toks[-1]
            generated_ids = list(prompt_toks)

            sys.stdout.write("Fluid Engine > " + prompt)
            sys.stdout.flush()

            last_fe = []
            tokens_emitted = 0
            for _ in range(20):
                if ponder_enabled:
                    probs, fe = engine.active_inference_step(curr_tok, ponder_steps=ponder_steps, reflex_strength=reflex_strength, dt=dt)
                    last_fe = fe
                else:
                    probs = engine.predict_distribution(curr_tok, dt=dt)

                next_tok = sample_word(probs, tokenizer, temperature=0.65, top_p=0.85, recent_tokens=generated_ids)
                generated_ids.append(next_tok)
                curr_tok = next_tok
                tokens_emitted += 1

                word = tokenizer.id2word.get(next_tok, "")
                if word in [".", ",", "!", "?", ":", ";"]:
                    sys.stdout.write(word)
                else:
                    sys.stdout.write(" " + word)
                sys.stdout.flush()

                if word == ".":
                    break

            gen_time = time.time() - t0
            tok_per_sec = tokens_emitted / max(gen_time, 1e-4)
            fe_info = f" | Free Energy: {' -> '.join([f'{x:.2f}' for x in last_fe])}" if (ponder_enabled and last_fe) else ""
            sys.stdout.write(f"\n[Stats: {tokens_emitted} words in {gen_time*1000:.1f} ms | {tok_per_sec:.1f} tok/s | RAM: {mem_mb:.2f} MB{fe_info}]\n")
            sys.stdout.flush()

        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")
            break

    engine.close()

if __name__ == "__main__":
    main()
