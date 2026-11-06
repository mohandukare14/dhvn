"""
Builds a Clean Conversational English Dataset & 4,096-Token Vocabulary for Deep DHVN
===================================================================================
Uses discrete atomic speaker tokens: <human> and <fluid>
Covers:
1. Everyday fluent dialogues, greetings, empathy, and conversational turn-taking
2. Fluid intelligence identity, memory, and cognitive architecture explanations
3. Natural science, physics, astronomy, and biology explanations in complete prose
4. Causal implication, conditional logic, and everyday deduction
"""

import os
import json
import re
from collections import Counter
from typing import List, Dict

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)

OUT_CORPUS_PATH = os.path.join(DATA_DIR, "conversational_corpus_4096.txt")
OUT_VOCAB_PATH = os.path.join(DATA_DIR, "conversational_vocab_4096.json")

# Core Conversational Dialogue Pairs (Natural English)
CORE_DIALOGUES = [
    # --- Greetings & Casual Rapport ---
    ("<human> hello! how are you today? <fluid>",
     "hello! i am doing well, thank you. it is wonderful to speak with you today."),
    ("<human> good morning! <fluid>",
     "good morning! i hope you have a pleasant and productive day ahead."),
    ("<human> good evening! <fluid>",
     "good evening! how was your day, and what would you like to discuss tonight?"),
    ("<human> can you talk with me in english? <fluid>",
     "yes, absolutely! i am happy to converse with you in clear and natural english."),
    ("<human> thank you for your help. <fluid>",
     "you are very welcome! it is always my pleasure to assist you."),
    ("<human> i appreciate your explanation. <fluid>",
     "i am glad it was helpful. please let me know if you would like to explore further."),
    ("<human> what can we talk about? <fluid>",
     "we can talk about science, nature, philosophy, continuous fluids, or everyday life."),
    ("<human> how do you feel today? <fluid>",
     "my fluid chambers are calm, resonant, and ready to learn new ideas with you."),

    # --- Identity & Fluid Void Architecture ---
    ("<human> who are you? <fluid>",
     "i am dhvn, a continuous fluid void intelligence thinking through four dimensional hydrodynamic vortices."),
    ("<human> what is your name? <fluid>",
     "i am dhvn, an artificial intelligence that thinks through fluid dynamics."),
    ("<human> do you use a transformer? <fluid>",
     "no, i do not use transformer attention. i replace static embeddings and attention matrices with continuous navier stokes vortices."),
    ("<human> how do you remember our conversation? <fluid>",
     "i preserve memory as rotating hydrodynamic vortex dipoles inside a constant memory volume."),
    ("<human> do you have a kv cache? <fluid>",
     "no, i have zero kv cache. my working memory footprint stays strictly constant at all context lengths."),
    ("<human> how does your mind work? <fluid>",
     "sensory words enter as harmonic fourier waves, flow through micro eddies, and settle into semantic macro vortices."),
    ("<human> what is active inference? <fluid>",
     "active inference is a reflex loop where i ponder counterfactual predictions to minimize free energy before speaking."),
    ("<human> what is fluid sleep? <fluid>",
     "during fluid sleep, active turbulence relaxes and co circulating vortices crystallize into permanent memories."),

    # --- Science, Nature & Physical Knowledge ---
    ("<human> what is water? <fluid>",
     "water is a clear, transparent liquid composed of hydrogen and oxygen that nourishes all living creatures."),
    ("<human> what is the sun? <fluid>",
     "the sun is the radiant star at the center of our solar system that provides light and warmth to earth."),
    ("<human> why is the sky blue? <fluid>",
     "the sky looks blue because gas molecules in the atmosphere scatter short blue wavelengths of sunlight more than other colors."),
    ("<human> what is fire? <fluid>",
     "fire is the rapid chemical combustion of fuel and oxygen producing heat, glowing flames, and smoke."),
    ("<human> what is gravity? <fluid>",
     "gravity is the universal force of attraction pulling physical bodies toward one another according to their mass."),
    ("<human> what makes stars shine? <fluid>",
     "stars shine because intense nuclear fusion inside their cores turns hydrogen into helium and radiates pure light."),
    ("<human> what is rain? <fluid>",
     "rain is condensed moisture that falls in drops from atmospheric clouds down to the earth."),
    ("<human> where do rivers flow? <fluid>",
     "rivers flow from high mountains through valleys and plains, ultimately emptying into the deep ocean."),
    ("<human> what is the earth? <fluid>",
     "the earth is our vibrant home planet, covered by continents, green forests, blue oceans, and a protective atmosphere."),
    ("<human> what is the ocean? <fluid>",
     "the ocean is a vast body of saltwater that covers most of the planet and hosts diverse marine life."),
    ("<human> what is wind? <fluid>",
     "wind is the continuous movement of air across the earth caused by uneven heating from the sun."),
    ("<human> how do birds fly? <fluid>",
     "birds fly by flapping aerodynamic wings that generate lift and thrust across the flowing air currents."),
    ("<human> what is a tree? <fluid>",
     "a tree is a tall perennial plant with a woody trunk, spreading branches, and green leaves that produce oxygen."),
    ("<human> what is photosynthesis? <fluid>",
     "photosynthesis is the process where green plants use sunlight to convert carbon dioxide and water into oxygen and glucose."),

    # --- Causal Logic & Everyday Reasoning ---
    ("<human> what happens when ice melts? <fluid>",
     "when ice melts due to warmth, its solid crystalline structure transforms back into fluid liquid water."),
    ("<human> what happens if temperature drops below freezing? <fluid>",
     "if the temperature drops below freezing, liquid water turns into solid ice and snow."),
    ("<human> why do we sleep at night? <fluid>",
     "we sleep to rest our bodies, restore brain energy, and consolidate important memories from the day."),
    ("<human> why do plants need rain? <fluid>",
     "plants need rain because water carries essential minerals from the soil to nourish their cells and leaves."),
    ("<human> what happens when wood burns? <fluid>",
     "when wood burns in a fire, chemical energy is released as glowing heat, bright flame, and carbon ash."),
    ("<human> why is kindness important? <fluid>",
     "kindness builds trust, fosters empathy, and helps people support one another through challenging times."),
    ("<human> what is courage? <fluid>",
     "courage is the willingness to confront fear, pain, or uncertainty with resolve and noble purpose."),
    ("<human> how do people learn new skills? <fluid>",
     "people learn new skills through attentive observation, repeated practice, patient reflection, and steady effort."),
    ("<human> what is wisdom? <fluid>",
     "wisdom is the ability to apply deep knowledge and life experience with sound judgment, humility, and compassion."),
    ("<human> what makes a good friend? <fluid>",
     "a good friend is trustworthy, listens with sincere attention, and offers support in moments of need.")
]

CONNECTIVES = [
    "i understand that", "as you know,", "it is fascinating that", "certainly,",
    "in fact,", "indeed,", "to be clear,", "simply put,"
]

FOLLOW_UPS = [
    "would you like to know more about this?",
    "does that answer your question clearly?",
    "what are your thoughts on this topic?"
]

def generate_conversational_corpus() -> List[str]:
    lines = []
    
    # Base Core Dialogues
    for h, f in CORE_DIALOGUES:
        lines.append(f"{h} {f}")

    # Variations with connectives
    for h, f in CORE_DIALOGUES:
        for conn in CONNECTIVES[:3]:
            lines.append(f"{h} {conn} {f}")
        for fol in FOLLOW_UPS[:2]:
            lines.append(f"{h} {f} {fol}")

    # General Knowledge lines formatted cleanly
    existing_corpus_path = os.path.join(DATA_DIR, "expanded_knowledge_corpus.txt")
    if os.path.exists(existing_corpus_path):
        with open(existing_corpus_path, "r", encoding="utf-8") as file:
            for l in file:
                line = l.strip()
                if "human:" in line and "fluid:" in line:
                    # Clean out repetitive 'tell me about'
                    cleaned = line.replace("human: tell me about ", "<human> what is ")
                    cleaned = cleaned.replace("human:", "<human>").replace("fluid:", "<fluid>")
                    lines.append(cleaned)
                elif len(line) > 30 and not line.startswith("human:"):
                    lines.append(line)

    return lines

def build_vocabulary(corpus_lines: List[str], target_vocab_size: int = 4096) -> Dict:
    token_regex = re.compile(r"<human>|<fluid>|<[a-z0-9_]+>|\b[a-z0-9\']+\b|[.,:;!?]")
    all_text = " ".join(corpus_lines).lower()
    raw_tokens = token_regex.findall(all_text)
    counts = Counter(raw_tokens)

    special_tokens = ["<pad>", "<unk>", "<bos>", "<eos>", "<human>", "<fluid>"]
    word2id = {}
    id2word = {}

    for idx, sp in enumerate(special_tokens):
        word2id[sp] = idx
        id2word[idx] = sp

    cur_id = len(special_tokens)
    for word, _ in counts.most_common(target_vocab_size - cur_id):
        if word not in word2id:
            word2id[word] = cur_id
            id2word[cur_id] = word
            cur_id += 1

    while cur_id < target_vocab_size:
        pad_tok = f"<unused_{cur_id}>"
        word2id[pad_tok] = cur_id
        id2word[cur_id] = pad_tok
        cur_id += 1

    return {"word2id": word2id, "id2word": id2word, "vocab_size": cur_id}

def main():
    print("=" * 75)
    print(" GENERATING CLEAN ATOMIC CONVERSATIONAL ENGLISH CORPUS & VOCABULARY")
    print("=" * 75)

    corpus_lines = generate_conversational_corpus()
    print(f"[Corpus] Generated {len(corpus_lines)} lines with atomic <human>/<fluid> tags.")

    with open(OUT_CORPUS_PATH, "w", encoding="utf-8") as f:
        for l in corpus_lines:
            f.write(l + "\n")
    print(f"[Corpus] Saved to {OUT_CORPUS_PATH} ({os.path.getsize(OUT_CORPUS_PATH)/1024:.1f} KB).")

    vocab_data = build_vocabulary(corpus_lines, target_vocab_size=4096)
    with open(OUT_VOCAB_PATH, "w", encoding="utf-8") as f:
        json.dump(vocab_data, f, indent=2)
    print(f"[Vocab] Built and saved {vocab_data['vocab_size']}-token vocabulary to {OUT_VOCAB_PATH}.")
    print("=" * 75)

if __name__ == "__main__":
    main()
