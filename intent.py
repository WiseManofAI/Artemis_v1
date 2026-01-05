def build_intent(thought):
    forecast = thought["forecast"]

    if forecast > 1.6:
        intent = "commit"
    elif forecast > 1.2:
        intent = "explore"
    elif forecast > 0.8:
        intent = "hold"
    else:
        intent = "withdraw"

    stance = (
        "optimistic" if forecast > 1.2 else
        "uncertain" if forecast > 0.8 else
        "guarded"
    )

    return {
        "emotion": thought["emotion"],
        "intent": intent,
        "stance": stance,
        "confidence": thought["certainty"]
    }
