import torch
import re
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_NAME = "unitary/toxic-bert"

print("Loading moderation model...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

model.eval()

labels = [
    "toxic",
    "severe_toxic",
    "obscene",
    "threat",
    "insult",
    "identity_hate"
]


def clean_text(text):

    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-Z ]", "", text)

    return text


def get_violation_type(results):

    if results["threat"] > 0.5:
        return "Threat"

    elif results["identity_hate"] > 0.5:
        return "Identity Hate"

    elif results["severe_toxic"] > 0.5:
        return "Severe Toxicity"

    elif results["insult"] > 0.4:
        return "Insult"

    elif results["obscene"] > 0.4:
        return "Obscene Language"

    elif results["toxic"] > 0.3:
        return "Toxic Language"

    else:
        return "None"


def predict(text):

    text = clean_text(text)

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    )

    with torch.no_grad():
        outputs = model(**inputs)

    probs = torch.sigmoid(outputs.logits).cpu().numpy()[0]

    results = dict(zip(labels, probs))

    severity = max(probs)

    violation_type = get_violation_type(results)

    if severity > 0.8:
        status = "Severe Policy Violation"

    elif severity > 0.5:
        status = "Policy Violation"

    elif results["toxic"] > 0.15 and results["threat"] > 0.08:
        status = "Potential Threat"

    else:
        status = "Safe Content"

    return results, severity, violation_type, status


if __name__ == "__main__":

    text = input("Enter text: ")

    results, severity, violation_type, status = predict(text)

    print("\nPrediction Scores")

    for k, v in results.items():
        print(f"{k}: {v:.2f}")

    print()

    if results["toxic"] > 0.3:
        print("Toxicity Detected: YES")
    else:
        print("Toxicity Detected: NO")

    print("Violation Type:", violation_type)

    print("\nSeverity Score:", round(severity, 2))

    print("Policy Status:", status)