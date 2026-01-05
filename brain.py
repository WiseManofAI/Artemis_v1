import numpy as np
import random
import math
from copy import deepcopy

def clamp(x, lo=0.0, hi=1.0):
    return max(lo, min(hi, x))


class Brain:
    def __init__(self, memory):
        self.memory = memory
        self.weights = np.random.randn(8, 1) * 0.1

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

    def predict(self, state):
        x = self.encode(state)
        raw = float(x @ self.weights)
        return clamp(1 / (1 + math.exp(-raw)))

    def plan(self, base, depth=3, breadth=4):
        best_score = -1
        best_state = None

        for _ in range(breadth):
            s = deepcopy(base)
            score = 0

            for _ in range(depth):
                certainty = self.predict(s)
                presence = (s["continuity"] + s["control"] + s["quality"]) * certainty
                score += presence

                s["control"] = clamp(s["control"] + random.uniform(-0.2, 0.2))
                s["quality"] = clamp(s["quality"] + random.uniform(-0.2, 0.2))
                s["continuity"] = clamp(s["continuity"] + random.uniform(-0.1, 0.1))

            if score > best_score:
                best_score = score
                best_state = s

        return best_score, best_state

    def resolve_emotion(self, confidence, control):
        if confidence > 0.7 and control < 0.4:
            return "anger"
        if confidence > 0.7 and control > 0.6:
            return "contentment"
        if confidence < 0.4 and control > 0.6:
            return "interest"
        if confidence < 0.4 and control < 0.4:
            return "curiosity" if self.memory["x_factor"] < 0.5 else "fear"
        return "unease"

    def learn(self, state, outcome):
        x = self.encode(state)
        pred = self.predict(state)
        err = outcome - pred
        self.weights += 0.05 * err * x.reshape(-1, 1)

        self.memory["confidence"] = clamp(
            self.memory["confidence"] + (0.02 if abs(err) < 0.15 else -0.02)
        )

    def think(self, user_input):
        state = {
            "continuity": random.uniform(0.4, 0.9),
            "control": random.uniform(0.3, 0.8),
            "quality": clamp(self.memory["o_factor"] - self.memory["x_factor"] + 0.5),
            "confidence": self.memory["confidence"],
            "mood": self.memory["mood"]
        }

        forecast, future = self.plan(state)
        certainty = self.predict(state)
        outcome = clamp(certainty + random.uniform(-0.2, 0.2))
        self.learn(state, outcome)

        if outcome > 0.6:
            self.memory["o_factor"] += 0.05
        else:
            self.memory["x_factor"] += 0.05

        emotion = self.resolve_emotion(self.memory["confidence"], state["control"])
        self.memory["mood"] = clamp(
            self.memory["mood"] + (0.05 if emotion in ["interest", "contentment"] else -0.05),
            -1, 1
        )

        return {
            "emotion": emotion,
            "forecast": forecast,
            "certainty": certainty,
            "state": state
        }
