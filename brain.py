import numpy as np
import random
import math
from copy import deepcopy

def clamp(x, lo=0.0, hi=1.0):
    return max(lo, min(hi, x))


class Brain:
    def __init__(self, memory):
        self.memory = memory

        # Predictor network
        self.weights = np.random.randn(8, 1) * 0.1

    # -------------------------
    # CONTEXT ENCODING
    # -------------------------
    def encode(self, state):
        return np.array([
            state["continuity"],
            state["control"],
            state["quality"],
            state["confidence"],
            state["mood"],
            self.memory["o_factor"],
            self.memory["x_factor"],
            random.random()
        ])

    # -------------------------
    # OUTCOME PREDICTION
    # -------------------------
    def predict_outcome(self, state):
        x = self.encode(state)
        raw = float(x @ self.weights)
        return clamp(1 / (1 + math.exp(-raw)))  # sigmoid certainty

    # -------------------------
    # MULTI-ACTION PLANNING
    # -------------------------
    def plan(self, base_state, depth=3, breadth=4):
        """
        Tree search over abstract actions.
        """
        best_path = None
        best_score = -1

        for _ in range(breadth):
            state = deepcopy(base_state)
            cumulative_presence = 0

            for _ in range(depth):
                certainty = self.predict_outcome(state)
                presence = (
                    state["continuity"]
                    + state["control"]
                    + state["quality"]
                ) * certainty

                cumulative_presence += presence

                # Simulate consequences
                state["control"] = clamp(state["control"] + random.uniform(-0.2, 0.2))
                state["quality"] = clamp(state["quality"] + random.uniform(-0.2, 0.2))
                state["continuity"] = clamp(state["continuity"] + random.uniform(-0.1, 0.1))

            if cumulative_presence > best_score:
                best_score = cumulative_presence
                best_path = state

        return best_score, best_path

    # -------------------------
    # EMOTION RESOLUTION
    # -------------------------
    def emotion(self, confidence, control):
        if confidence > 0.7 and control < 0.3:
            return "anger"
        if confidence > 0.7 and control > 0.6:
            return "contentment"
        if confidence < 0.4 and control > 0.6:
            return "interest"
        if confidence < 0.4 and control < 0.4:
            return "curiosity" if self.memory["x_factor"] < 0.5 else "fear"
        return "unease"

    # -------------------------
    # LEARNING UPDATE
    # -------------------------
    def learn(self, state, outcome):
        x = self.encode(state)
        pred = self.predict_outcome(state)
        error = outcome - pred
        self.weights += 0.05 * error * x.reshape(-1, 1)

        # Confidence update
        if abs(error) < 0.15:
            self.memory["confidence"] = clamp(self.memory["confidence"] + 0.02)
        else:
            self.memory["confidence"] = clamp(self.memory["confidence"] - 0.02)

    # -------------------------
    # THOUGHT → LANGUAGE
    # -------------------------
    def verbalize(self, state, emotion, forecast):
        """
        Organic language from internal variables.
        """
        tone = []
        if emotion in ["interest", "curiosity"]:
            tone.append("exploring")
        if emotion in ["anger", "fear"]:
            tone.append("guarded")
        if emotion in ["contentment"]:
            tone.append("assured")

        sentence = (
            f"I am {', '.join(tone)} this. "
            f"My sense of control feels {state['control']:.2f}, "
            f"and the future presence trajectory looks {forecast:.2f}. "
            f"I’m adjusting my stance as I process what this could become."
        )
        return sentence

    # -------------------------
    # MAIN THINK LOOP
    # -------------------------
    def think(self, user_input):
        state = {
            "continuity": random.uniform(0.4, 0.9),
            "control": random.uniform(0.3, 0.8),
            "quality": clamp(self.memory["o_factor"] - self.memory["x_factor"] + 0.5),
            "confidence": self.memory["confidence"],
            "mood": self.memory["mood"]
        }

        forecast, future_state = self.plan(state)
        certainty = self.predict_outcome(state)

        # Outcome simulation
        outcome = clamp(certainty + random.uniform(-0.2, 0.2))
        self.learn(state, outcome)

        # O/X updates
        if outcome > 0.6:
            self.memory["o_factor"] += 0.1
        else:
            self.memory["x_factor"] += 0.1

        emotion = self.emotion(
            self.memory["confidence"],
            state["control"]
        )

        # Mood update (emotion-driven)
        if emotion in ["contentment", "interest"]:
            self.memory["mood"] += 0.05
        elif emotion in ["anger", "fear"]:
            self.memory["mood"] -= 0.05

        self.memory["mood"] = clamp(self.memory["mood"], -1, 1)

        return self.verbalize(state, emotion, forecast), emotion, forecast
