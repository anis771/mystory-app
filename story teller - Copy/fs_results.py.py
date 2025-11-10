import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

import json

# Download NLTK data (run once)
nltk.download("punkt")
nltk.download('vader_lexicon')
# Initialize once
sentiment_analyzer = SentimentIntensityAnalyzer()


def extract_paragraph_texts(json_path):
    """
    Reads a JSON file and extracts all 'text' values from the 'paragraphs' list.

    Args:
        json_path (str): Path to the JSON file.

    Returns:
        list: A list of paragraph texts.
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Check if 'paragraphs' exists and extract 'text'
    if "paragraphs" not in data:
        raise KeyError("The JSON file does not contain a 'paragraphs' key.")

    texts = [p["text"] for p in data["paragraphs"] if "text" in p]
    return texts

def analyze_sentiment(text: str) -> dict:
    """
    Perform sentiment analysis on text using NLTK's VADER.
    Returns compound, pos, neu, neg scores.
    """
    if not text or not isinstance(text, str):
        return {"compound": 0.0, "pos": 0.0, "neu": 0.0, "neg": 0.0}
    
    scores = sentiment_analyzer.polarity_scores(text)
    return scores


if __name__ == "__main__":
    file_path = "experiments/exp_20250901_164827.json"
    paragraphs = extract_paragraph_texts(file_path)
    
    print("Extracted Paragraphs:")
    
    for i, text in enumerate(paragraphs, start=1):
        print(f"\nParagraph {i}:\n{text}\n")
        sentiment_scores = analyze_sentiment(text)
        print(f"sentiment score {sentiment_scores}")