"""
Generates a comprehensive English dictionary and vocabulary dataset for DHVN.
Includes:
1. Core frequency vocabulary (nouns, verbs, adjectives, adverbs, connectives).
2. Lexical definitions of fundamental concepts.
3. Morphological inflections and grammatical conjugations.
4. Semantic associations and antonym/synonym pairs.
"""

import os

def create_dictionary():
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "english_dictionary.txt")

    entries = []

    # 1. Fundamental Lexical Definitions
    definitions = [
        ("water", "a clear liquid that falls as rain and forms rivers, lakes, and oceans"),
        ("fire", "the combustion of substances producing light, heat, and flame"),
        ("earth", "the planet on which we live, or the solid surface of the world"),
        ("air", "the invisible gaseous substance surrounding the earth"),
        ("sun", "the star around which the earth orbits, providing light and heat"),
        ("moon", "the natural satellite of the earth, visible by reflected sunlight"),
        ("star", "a luminous celestial body of burning gas visible in the night sky"),
        ("sky", "the region of the atmosphere and outer space seen from the earth"),
        ("tree", "a woody perennial plant with a single stem or trunk and branches"),
        ("forest", "a large area covered chiefly with trees and undergrowth"),
        ("river", "a large natural stream of water flowing in a channel to the sea"),
        ("ocean", "a very large expanse of sea, in particular each of the main areas"),
        ("mountain", "a large natural elevation of the earth surface rising abruptly"),
        ("valley", "a low area of land between hills or mountains, often with a river"),
        ("wind", "the perceptible natural movement of the air blowing in a direction"),
        ("cloud", "a visible mass of condensed water vapor floating in the atmosphere"),
        ("rain", "moisture condensed from the atmosphere that falls in drops"),
        ("bird", "a feathered vertebrate animal with wings and a beak that can fly"),
        ("fish", "a limbless cold-blooded animal with gills and fins living in water"),
        ("flower", "the seed-bearing part of a plant consisting of reproductive organs"),
        ("apple", "the round fruit of a tree with red, green, or yellow skin"),
        ("bread", "food made of flour, water, and yeast or another leavening agent"),
        ("house", "a building for human habitation, especially one that is lived in"),
        ("home", "the place where one lives permanently, especially as a family"),
        ("friend", "a person whom one knows and with whom one has a bond of affection"),
        ("child", "a young human being below the age of puberty or legal majority"),
        ("man", "an adult human male individual"),
        ("woman", "an adult human female individual"),
        ("time", "the continuous progress of existence and events in the past, present, future"),
        ("space", "the continuous physical expanse that is boundless in all directions"),
        ("light", "the natural agent that stimulates sight and makes things visible"),
        ("dark", "with little or no light, or the absence of illumination"),
        ("thought", "an idea or opinion produced by thinking or occurring suddenly in the mind"),
        ("mind", "the element of a person that enables them to be aware of the world"),
        ("heart", "the organ that pumps blood, or the center of emotion and feeling"),
        ("hand", "the end part of a person arm beyond the wrist, including fingers"),
        ("book", "a written or printed work consisting of pages glued or sewn together"),
        ("word", "a single distinct meaningful element of speech or writing"),
        ("language", "the method of human communication, either spoken or written"),
        ("knowledge", "facts, information, and skills acquired through experience or education"),
        ("wisdom", "the quality of having experience, knowledge, and good judgment"),
        ("peace", "freedom from disturbance, or a state of tranquility and calm"),
        ("love", "an intense feeling of deep affection and care for another"),
        ("truth", "the quality or state of being in accord with fact or reality"),
        ("beauty", "a combination of qualities that pleases the aesthetic senses"),
        ("courage", "the ability to do something that frightens one, or bravery"),
        ("freedom", "the power or right to act, speak, or think as one wants"),
        ("vortex", "a mass of whirling fluid or air, especially a whirlpool or whirlwind"),
        ("fluid", "a substance that has no fixed shape and yields easily to external pressure"),
        ("motion", "the action or process of moving or being moved across space")
    ]

    for word, defn in definitions:
        entries.append(f"{word}: {defn}.\n")

    # 2. Comprehensive Vocabulary Categorical Lists
    vocabulary_categories = {
        "Adjectives": [
            "good", "bad", "great", "small", "large", "tiny", "huge", "vast", "deep",
            "shallow", "high", "low", "long", "short", "wide", "narrow", "bright",
            "dark", "soft", "hard", "heavy", "light", "warm", "cold", "hot", "cool",
            "swift", "slow", "fast", "calm", "rough", "smooth", "clear", "cloudy",
            "sweet", "sour", "bitter", "fresh", "stale", "pure", "golden", "silver",
            "green", "blue", "red", "yellow", "white", "black", "purple", "silent",
            "loud", "peaceful", "wild", "gentle", "fierce", "wise", "clever", "foolish",
            "happy", "sad", "joyful", "brave", "honest", "kind", "cruel", "noble"
        ],
        "Verbs": [
            "be", "have", "do", "say", "go", "get", "make", "know", "think", "take",
            "see", "come", "want", "look", "use", "find", "give", "tell", "work",
            "call", "try", "ask", "need", "feel", "become", "leave", "put", "mean",
            "keep", "let", "begin", "seem", "help", "talk", "turn", "start", "show",
            "hear", "play", "run", "move", "like", "live", "believe", "hold", "bring",
            "happen", "write", "provide", "sit", "stand", "lose", "pay", "meet",
            "include", "continue", "set", "learn", "change", "lead", "understand",
            "watch", "follow", "stop", "create", "speak", "read", "allow", "add",
            "spend", "grow", "open", "walk", "win", "offer", "remember", "love",
            "consider", "appear", "buy", "wait", "serve", "die", "send", "expect",
            "build", "stay", "fall", "cut", "reach", "kill", "remain", "flow", "soar",
            "fly", "whisper", "shine", "glow", "gather", "scatter", "unfold", "dissolve"
        ],
        "Nouns": [
            "time", "year", "people", "way", "day", "man", "thing", "woman", "life",
            "child", "world", "school", "state", "family", "student", "group", "country",
            "problem", "hand", "part", "place", "case", "week", "company", "system",
            "program", "question", "work", "government", "number", "night", "point",
            "home", "water", "room", "mother", "area", "money", "story", "fact",
            "month", "lot", "right", "study", "book", "eye", "job", "word", "business",
            "issue", "side", "kind", "head", "house", "service", "friend", "father",
            "power", "hour", "game", "line", "end", "member", "law", "car", "city",
            "community", "name", "president", "team", "minute", "idea", "kid", "body",
            "information", "back", "parent", "face", "others", "level", "office", "door",
            "health", "person", "art", "war", "history", "party", "result", "change",
            "morning", "reason", "research", "girl", "guy", "moment", "air", "teacher",
            "force", "education", "ocean", "river", "mountain", "forest", "tree", "bird",
            "flower", "stone", "cloud", "wind", "shadow", "flame", "sound", "silence",
            "path", "valley", "horizon", "kingdom", "palace", "village", "bridge", "gate"
        ]
    }

    for cat, words in vocabulary_categories.items():
        entries.append(f"\n# Category: {cat}\n")
        chunk = []
        for i, w in enumerate(words):
            chunk.append(w)
            if len(chunk) == 8 or i == len(words) - 1:
                entries.append(" ".join(chunk) + "\n")
                chunk = []

    # 3. Morphological Grammar Paradigms
    conjugations = [
        ("walk", "walks", "walking", "walked"),
        ("run", "runs", "running", "ran"),
        ("fly", "flies", "flying", "flew"),
        ("flow", "flows", "flowing", "flowed"),
        ("sing", "sings", "singing", "sang"),
        ("see", "sees", "seeing", "saw"),
        ("think", "thinks", "thinking", "thought"),
        ("know", "knows", "knowing", "knew"),
        ("speak", "speaks", "speaking", "spoke"),
        ("write", "writes", "writing", "wrote"),
        ("shine", "shines", "shining", "shone"),
        ("grow", "grows", "growing", "grew"),
        ("build", "builds", "building", "built"),
        ("find", "finds", "finding", "found"),
        ("give", "gives", "giving", "gave"),
        ("take", "takes", "taking", "took"),
        ("bring", "brings", "bringing", "brought"),
        ("tell", "tells", "telling", "told"),
        ("hear", "hears", "hearing", "heard"),
        ("feel", "feels", "feeling", "felt")
    ]

    entries.append("\n# Morphological Conjugations (Base, Present, Participle, Past)\n")
    for base, pres, part, past in conjugations:
        entries.append(f"{base} -> {pres} -> {part} -> {past}.\n")

    # 4. Semantic Relationships (Synonyms and Antonyms)
    relations = [
        ("light", "opposite of", "dark"),
        ("day", "opposite of", "night"),
        ("hot", "opposite of", "cold"),
        ("high", "opposite of", "low"),
        ("good", "opposite of", "bad"),
        ("fast", "opposite of", "slow"),
        ("true", "opposite of", "false"),
        ("sweet", "opposite of", "bitter"),
        ("huge", "synonym of", "large"),
        ("tiny", "synonym of", "small"),
        ("swift", "synonym of", "fast"),
        ("brave", "synonym of", "courageous"),
        ("clever", "synonym of", "wise"),
        ("joyful", "synonym of", "happy"),
        ("gentle", "synonym of", "peaceful"),
        ("boundless", "synonym of", "infinite")
    ]

    entries.append("\n# Semantic Associations\n")
    for w1, rel, w2 in relations:
        entries.append(f"{w1} is the {rel} {w2}.\n")

    full_dict_text = "".join(entries)

    # Repeat definitions in structured narrative sentences for deep contextual grounding
    sentences = [
        "The clear river flows through the green forest into the deep ocean.",
        "A little bird flies high into the bright blue sky above the mountain.",
        "The golden sun shines warmly on the valley, bringing light to all living things.",
        "Wise people seek knowledge, truth, and peace through patient learning.",
        "Water and air are essential fluids that move across the surface of the earth.",
        "A gentle wind whispers softly through the branches of the ancient oak tree.",
        "Every word in human language carries thought, meaning, and feeling across time."
    ]

    context_text = "\n\n# Contextual Syntactic Structures\n" + "\n".join(sentences * 15)
    full_dict_text += context_text

    with open(out_path, "w", encoding="ascii", errors="ignore") as f:
        f.write(full_dict_text)

    size_kb = len(full_dict_text.encode('ascii')) / 1024.0
    print(f"Created English Dictionary: {out_path} ({size_kb:.1f} KB, {len(full_dict_text.split())} words)")

if __name__ == "__main__":
    create_dictionary()

