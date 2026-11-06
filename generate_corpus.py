import os
import random

def generate_expanded_corpus():
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "expanded_corpus.txt")

    subjects = [
        "The little bird", "The brave knight", "A clever fox", "The quiet rabbit",
        "A wise old owl", "The little girl named Lily", "The golden fish", "A friendly bear",
        "The gentle giant", "A playful puppy", "The young prince", "The honest woodcutter"
    ]
    
    actions = [
        "lived near a green forest and loved to sing sweet songs",
        "walked across the high mountain under the golden sun",
        "found a secret garden filled with red flowers and fresh apples",
        "hopped along the river bank watching the clear water flow",
        "built a small wooden house with a warm fireplace inside",
        "saw a shiny golden key hidden beneath the green moss",
        "helped a little lost friend find the way back home",
        "searched for the magic stone that glowed in the dark night",
        "rested under a tall oak tree while the cool wind blew softly",
        "shared delicious bread and sweet honey with everyone in the village"
    ]

    events = [
        "One sunny morning, the sky was bright and blue without a single cloud.",
        "Suddenly, a strange sound echoed through the deep quiet woods.",
        "The kind queen smiled warmly and offered a gift of pure gold.",
        "They walked together through the meadow, laughing and telling happy stories.",
        "As the silver moon rose into the dark night sky, all the stars began to sparkle.",
        "The river flowed quietly to the deep blue sea, carrying golden leaves on its surface.",
        "Everyone in the great kingdom celebrated with music, dancing, and joy.",
        "The cold winter wind began to blow, bringing soft white snow to the quiet valley.",
        "Deep inside the ancient castle, a warm candle flickered on the stone table.",
        "With a joyful heart, the little traveler knew that true friendship was the greatest treasure."
    ]

    dialogues = [
        '"Do not be afraid," whispered the gentle friend. "I will help you find the way."',
        '"Look at that beautiful castle on the hill!" exclaimed the excited child.',
        '"Thank you for your kindness," said the old traveler with a peaceful smile.',
        '"We must cross the wooden bridge before the sun sets behind the distant mountain."',
        '"Every good deed brings light into the world," replied the wise king gently.'
    ]

    paragraphs = []
    rng = random.Random(1337)

    # Generate ~600 diverse, grammatically sound narrative paragraphs
    for _ in range(600):
        sub = rng.choice(subjects)
        act = rng.choice(actions)
        evt = rng.choice(events)
        p = f"{sub} {act}. {evt} "
        if rng.random() > 0.4:
            p += rng.choice(dialogues) + " "
        if rng.random() > 0.3:
            sub2 = rng.choice(subjects)
            act2 = rng.choice(actions)
            p += f"{sub2} {act2}."
        paragraphs.append(p.strip())

    full_text = "\n\n".join(paragraphs)
    
    with open(out_path, "w", encoding="ascii", errors="ignore") as f:
        f.write(full_text)

    size_kb = len(full_text.encode('ascii')) / 1024.0
    print(f"Generated expanded corpus: {out_path} ({size_kb:.1f} KB, {len(paragraphs)} paragraphs, {len(full_text.split())} words)")

if __name__ == "__main__":
    generate_expanded_corpus()

