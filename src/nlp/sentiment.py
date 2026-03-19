from __future__ import annotations

import pandas as pd

try:
    import nltk
    from nltk.sentiment import SentimentIntensityAnalyzer
except Exception:  # pragma: no cover
    nltk = None
    SentimentIntensityAnalyzer = None


class SimpleSentimentEngine:
    def __init__(self) -> None:
        self.analyzer = None
        if SentimentIntensityAnalyzer is not None and nltk is not None:
            try:
                self.analyzer = SentimentIntensityAnalyzer()
            except LookupError:
                nltk.download("vader_lexicon", quiet=True)
                self.analyzer = SentimentIntensityAnalyzer()

    def score_text(self, text: str) -> float:
        if not isinstance(text, str):
            text = ""

        if self.analyzer is not None:
            return float(self.analyzer.polarity_scores(text).get("compound", 0.0))

        positive_words = {"confirmed", "thank", "on time", "attend"}
        negative_words = {"delay", "difficult", "miss", "reschedule", "not sure"}
        lower_text = text.lower()
        score = 0.0
        for word in positive_words:
            if word in lower_text:
                score += 0.2
        for word in negative_words:
            if word in lower_text:
                score -= 0.2
        return max(min(score, 1.0), -1.0)


def label_sentiment(score: float) -> str:
    if score >= 0.2:
        return "positive"
    if score <= -0.2:
        return "negative"
    return "neutral"


def enrich_with_sentiment(df: pd.DataFrame, text_col: str = "patient_feedback_text") -> pd.DataFrame:
    output = df.copy()
    engine = SimpleSentimentEngine()

    output["sentiment_score"] = output[text_col].fillna("").apply(engine.score_text)
    output["sentiment_label"] = output["sentiment_score"].apply(label_sentiment)
    return output
