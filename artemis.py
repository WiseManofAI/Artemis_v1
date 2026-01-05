import json
import sys
import os
from brain import Brain
from intent import build_intent
from language_llm import render

MEMORY_FILE = "memory.json"

def load_memory():
    if not os.path.exists(MEMORY_FILE):
        return {
            "confidence": 0.4,
            "mood": 0.0,
            "o_factor": 0.0,
            "x_factor": 0.0
        }
    with open(MEMORY_FILE, "r") as f:
        return json.load(f)

def save_memory(mem):
    with open(MEMORY_FILE, "w") as f:
        json.dump(mem, f, indent=2)

def main():
    memory = load_memory()
    brain = Brain(memory)

    print("Artemis online.")
    print("Commands: Die | Die:Re | Stats\n")

    while True:
        user = input("You > ").strip()

        if user == "Die":
            print("Artemis terminated.")
            sys.exit(0)

        if user == "Die:Re":
            memory.update({
                "confidence": 0.4,
                "mood": 0.0,
                "o_factor": 0.0,
                "x_factor": 0.0
            })
            save_memory(memory)
            print("Artemis reset complete.")
            continue

        if user == "Stats":
            print(json.dumps(memory, indent=2))
            continue

        thought = brain.think(user)
        intent = build_intent(thought)
        response = render(intent)

        save_memory(memory)
        print(f"Artemis > {response}\n")

if __name__ == "__main__":
    main()
