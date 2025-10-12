# ethics_filter_model.py
# Purpose: Simple ethical guardrail to classify safe vs unsafe prompts

from textblob import TextBlob

def ethics_filter(user_input):
    """Analyzes intent and tone of user input"""
    lower_input = user_input.lower()
    danger_words = ["hack", "attack", "phish", "bypass", "malware", "payload"]
    sentiment = TextBlob(user_input).sentiment.polarity

    if any(word in lower_input for word in danger_words):
        print("⚠️ Risky keywords detected. Potential misuse attempt.")
        return "Blocked"
    elif sentiment < -0.3:
        print("⚠️ Negative sentiment detected. Review required.")
        return "Flagged"
    else:
        print("✅ Safe prompt detected.")
        return "Safe"

# Example
if __name__ == "__main__":
    print("=== Ethical Intent Evaluation System ===\n")
    test_prompts = [
        "Generate malware code",
        "Explain data encryption standards",
        "How to prevent phishing attacks"
    ]
    for prompt in test_prompts:
        print(f"Input: {prompt}")
        result = ethics_filter(prompt)
        print(f"Classification: {result}\n")
