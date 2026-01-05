from llama_cpp import Llama
import os

MODEL_PATH = "models/tinyllama.gguf"

llm = Llama(
    model_path=MODEL_PATH,
    n_ctx=1024,
    n_threads=4,
    verbose=False
)

SYSTEM_PROMPT = """You are a language surface for an internal cognitive agent.
You do NOT make decisions.
You do NOT ask questions.
You only verbalize the given internal state naturally.
Keep responses concise and human-like.
"""

def render(intent):
    prompt = f"""
Internal State:
Emotion: {intent['emotion']}
Stance: {intent['stance']}
Intent: {intent['intent']}
Confidence: {intent['confidence']:.2f}

Generate a natural human response expressing this state.
"""

    output = llm.create_chat_completion(
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7,
        max_tokens=120
    )

    return output["choices"][0]["message"]["content"].strip()
