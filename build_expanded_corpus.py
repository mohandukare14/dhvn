"""
Corpus Generator for DHVN 4,096-Token Vocabulary Expansion
Creates a comprehensive multi-domain knowledge corpus covering science, philosophy,
history, ecology, logic, fables, and structured reasoning.
"""

import os

CORPUS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "expanded_knowledge_corpus.txt")

domains = {
    "physics_and_nature": [
        "energy cannot be created or destroyed, it only transforms from one state to another.",
        "gravity is the attractive force that pulls objects toward the center of massive bodies.",
        "light travels across empty space as both a particle and an electromagnetic wave.",
        "sound requires a physical medium like air, water, or solid stone to propagate vibrations.",
        "heat is the kinetic energy of vibrating molecules moving within a substance.",
        "pressure increases with depth because the weight of fluid above exerts continuous force.",
        "momentum is the product of mass and velocity, conserved in all closed physical systems.",
        "friction opposes the relative motion of surfaces in contact, converting motion into heat.",
        "inertia is the natural resistance of any physical object to a change in its velocity.",
        "electricity is the flow of electric charge through conductive materials like copper.",
        "magnetism arises from moving electric charges and intrinsic magnetic moments of particles.",
        "density is the amount of mass contained within a given unit volume of matter.",
        "buoyancy allows lighter fluids or hollow objects to float atop denser liquid volumes.",
        "temperature measures the average kinetic energy of the particles within an object.",
        "waves carry energy through space without permanently transporting physical matter.",
        "refraction bends the path of light when it passes from one transparent medium to another.",
        "reflection occurs when wave fronts bounce back from a boundary surface without entering.",
        "resonance happens when an oscillating system is driven at its natural harmonic frequency.",
        "viscosity measures the internal resistance of a fluid to shear deformation and flow.",
        "turbulence consists of chaotic fluid motion characterized by swirling eddies and vortices.",
        "vorticity represents the local spinning motion of a fluid continuum near a given point.",
        "circulation measures the net rotational flux of velocity along a closed fluid loop.",
        "conservation laws govern the balance of mass, momentum, and angular circulation in nature."
    ],
    "astronomy_and_space": [
        "the sun is a luminous yellow star providing continuous light and warmth to our solar system.",
        "planets orbit the sun along elliptical paths governed by gravitational attraction.",
        "the moon orbits the earth, creating ocean tides through its gravitational pull.",
        "stars generate heat and radiant energy through nuclear fusion in their dense cores.",
        "galaxies are vast collections of billions of stars, interstellar dust, and gas held by gravity.",
        "comets are icy bodies that release glowing tails of vapor when approaching the warm sun.",
        "asteroids are rocky remnants left over from the early formation of planetary systems.",
        "the atmosphere shields the surface of the earth from harmful cosmic rays and meteoroids.",
        "outer space is an expansive vacuum containing very low density of particles and radiation.",
        "day and night alternate as the spherical earth rotates upon its tilted polar axis.",
        "seasons change because the earth tilts as it journeys along its annual orbit around the sun.",
        "the night sky displays distant constellations that guided navigators across ancient seas.",
        "telescopes gather faint photons across vast distances to reveal the deep universe."
    ],
    "biology_and_ecology": [
        "all living organisms are composed of cells that metabolize nutrients and reproduce.",
        "plants convert sunlight into chemical energy through the process of photosynthesis.",
        "green leaves contain chlorophyll that absorbs radiant light and produces oxygen.",
        "water is essential for all known biological life to transport nutrients and maintain cells.",
        "forests act as the green lungs of our planet, absorbing carbon and releasing pure air.",
        "rivers provide freshwater that sustains animals, vegetation, and human communities.",
        "soil is rich with organic matter, minerals, and microscopic organisms that nurture roots.",
        "seeds store genetic information and nourishment until warmth and moisture induce germination.",
        "flowers bloom in spring, attracting pollinating insects with sweet nectar and color.",
        "birds have lightweight hollow bones and aerodynamic feathers that enable soaring flight.",
        "fish breathe dissolved oxygen through specialized gills while swimming with flexible fins.",
        "predators and prey maintain ecological equilibrium through natural food chains.",
        "ecosystems depend on the harmonious interaction of climate, water, soil, and living species."
    ],
    "philosophy_and_mind": [
        "wisdom begins with recognizing the limits of one's own knowledge and understanding.",
        "consciousness is the subjective experience of awareness, perception, and thought.",
        "reasoning is the deliberate process of drawing sound conclusions from reliable premises.",
        "truth is the correspondence between mental propositions and objective reality.",
        "memory preserves past impressions so the mind can anticipate future consequences.",
        "time flows continuously forward, linking causes in the past to effects in the present.",
        "knowledge grows through disciplined observation, experimentation, and critical reflection.",
        "ethics guides human conduct toward justice, benevolence, honesty, and mutual respect.",
        "patience allows clarity to emerge from the turbulent confusion of raw emotional impulses.",
        "a peaceful mind observes the world like a calm lake reflecting the clear blue sky.",
        "ideas are like seeds that blossom into creative actions when nurtured by steady effort.",
        "language gives structure to thought, enabling minds to share meaning across generations.",
        "understanding requires listening with empathy and examining ideas from multiple perspectives."
    ],
    "technology_and_machines": [
        "computers process information by executing sequences of logical instructions at high speed.",
        "algorithms are step-by-step procedures designed to solve specific computational problems.",
        "memory storage records data in binary patterns that can be retrieved and transformed.",
        "engines convert chemical or electrical energy into mechanical work and motion.",
        "bridges span rivers and deep valleys through balanced distribution of tensile and compressive forces.",
        "networks connect distant devices to exchange signals, messages, and collaborative knowledge.",
        "tools amplify human capability, allowing the construction of cities, roads, and vessels.",
        "sensors detect changes in temperature, pressure, light, and motion to inform controllers.",
        "efficiency measures the ratio of useful energy output produced relative to the total energy input."
    ],
    "causality_and_logic": [
        "if temperature drops below freezing, water solidifies into crystalline ice.",
        "when rain falls upon dry soil, seeds awaken and green shoots emerge toward the sunlight.",
        "because the river flows downhill, it carves channels through solid rock over millennia.",
        "if an object experiences balanced forces, it maintains constant velocity without acceleration.",
        "when fire contacts dry wood, combustion produces glowing flames, heat, and smoke.",
        "because sunlight warms the surface, atmospheric air expands, rises, and creates winds.",
        "if a person studies diligently, their understanding of complex concepts deepens over time.",
        "when clouds condense into heavy droplets, precipitation falls upon the awaiting earth.",
        "because friends support one another during hardship, their mutual bond grows resilient.",
        "if you cultivate virtuous habits, your actions naturally reflect integrity and kindness."
    ],
    "questions_and_answers": [
        "question: what is water? answer: water is a clear liquid that falls as rain and forms rivers, lakes, and oceans.",
        "question: what is the sun? answer: the sun is the radiant star providing light and heat to our planet.",
        "question: what is fire? answer: fire is the combustion of matter producing heat, light, and rising flame.",
        "question: what is a river? answer: a river is a natural stream of freshwater flowing to the ocean.",
        "question: what is a mountain? answer: a mountain is a high natural elevation of the earth surface rising steeply.",
        "question: what is the earth? answer: the earth is our planetary home with oceans, atmosphere, and diverse life.",
        "question: what is a forest? answer: a forest is an expansive territory dominated by trees and rich undergrowth.",
        "question: what is a star? answer: a star is a luminous celestial sphere of burning gas visible in the night sky.",
        "question: what is wind? answer: wind is the perceptible horizontal movement of air across the earth surface.",
        "question: what is a tree? answer: a tree is a perennial woody plant with a single trunk, branches, and leaves.",
        "question: what is a friend? answer: a friend is a trusted companion with whom one shares mutual goodwill.",
        "question: what is learning? answer: learning is the active acquisition of knowledge, skill, and insight.",
        "question: who are you? answer: i am a continuous 4d fluid void intelligence thinking through hydrodynamic vortices.",
        "question: how do you store memory? answer: memory is preserved as circulating vortices in a constant volume void.",
        "question: where does knowledge come from? answer: knowledge emerges from systematic observation and deep reflection."
    ],
    "fables_and_tales": [
        "the wise old owl perched quietly upon the ancient oak tree, watching the silver river flow.",
        "a gentle breeze whispered through the golden valley as travelers rested beneath warm sunlight.",
        "the courageous voyager sailed into the open ocean, guided solely by the constant northern star.",
        "deep within the quiet forest, clear water bubbled from a hidden spring among mossy stones.",
        "the golden bird sang a sweet melody at dawn, awakening the sleeping village with gladness.",
        "high upon the rocky summit, the eagle spread wide wings and drifted upon rising thermal currents.",
        "children gathered around the warm hearth fire to hear tales of distant lands and ancient heroes.",
        "a small stream began high among the mountain snows, growing stronger until it reached the wide sea."
    ]
}

def generate_corpus():
    os.makedirs(os.path.dirname(CORPUS_PATH), exist_ok=True)
    all_lines = []
    
    # 1. Base domain sentences
    for cat, lines in domains.items():
        all_lines.extend(lines)
        
    # 2. Permutations, compound logic, and variations to reach 2,500+ rich lines
    variations = []
    for line in all_lines[:]:
        variations.append(line)
        # Create dialog variant
        if not line.startswith("question:"):
            variations.append(f"human: tell me about {line.split()[0]}. fluid: {line}")
            variations.append(f"in truth, {line}")
            variations.append(f"we observe that {line}")

    # Replicate with contextual variations
    final_corpus = []
    for v in variations:
        final_corpus.append(v)

    # Add cross-domain comparative sentences
    comparatives = [
        "just as a river carves a stone through persistence, the mind masters difficult knowledge through patient study.",
        "like stars illuminating the night sky, wise insights bring light to dark and uncertain situations.",
        "as roots anchor a majestic tree against strong storms, sound principles anchor character through turbulent times.",
        "fluid flows around solid obstacles, demonstrating that adaptability is often stronger than rigid force.",
        "the calm surface of a mountain lake reveals the depth beneath, just as quiet contemplation reveals truth.",
        "a single spark can ignite a grand fire, just as a single novel idea can inspire generations of discovery.",
        "seeds buried deep in cold earth do not despair; they patiently wait for spring warmth to blossom.",
        "rivers never rush backward to the source; they move steadily forward toward their union with the ocean.",
        "the wind cannot be held in human hands, yet its immense power turns great windmills and shapes dunes.",
        "knowledge without compassion is like a winter sun, providing bright light but little warming comfort."
    ]
    for c in comparatives:
        for _ in range(5):
            final_corpus.append(c)

    # Repeat core sentences to build robust frequency distribution
    multiplied_corpus = []
    for _ in range(12):
        for line in final_corpus:
            multiplied_corpus.append(line)

    with open(CORPUS_PATH, "w", encoding="utf-8") as f:
        for line in multiplied_corpus:
            f.write(line + "\n")

    print(f"[Corpus] Generated {len(multiplied_corpus)} lines at {CORPUS_PATH}")
    file_size_kb = os.path.getsize(CORPUS_PATH) / 1024.0
    print(f"[Corpus] File size: {file_size_kb:.1f} KB")

if __name__ == "__main__":
    generate_corpus()

