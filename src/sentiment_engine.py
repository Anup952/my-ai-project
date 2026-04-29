"""Sentiment and emotion analysis functions."""

from __future__ import annotations

from collections import Counter
from functools import lru_cache
import re

from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

try:
    from nrclex import NRCLex

    NRCLEX_AVAILABLE = True
except ImportError:  # pragma: no cover - handled gracefully at runtime
    NRCLex = None
    NRCLEX_AVAILABLE = False


FALLBACK_EMOTION_LEXICON = {
    "joy": [
        "happy",
        "happiness",
        "joy",
        "joyful",
        "great",
        "excited",
        "excellent",
        "love",
        "loved",
        "like",
        "liked",
        "wonderful",
        "amazing",
        "awesome",
        "fantastic",
        "best",
        "positive",
        "delight",
        "smile",
        "pleased",
        "satisfied",
        "good",
    ],
    "sadness": [
        "sad",
        "sadness",
        "upset",
        "depressed",
        "cry",
        "cried",
        "crying",
        "heartbroken",
        "lonely",
        "miserable",
        "gloomy",
        "unhappy",
        "hurt",
        "disappointed",
        "disappointing",
        "bad",
    ],
    "anger": [
        "angry",
        "anger",
        "furious",
        "mad",
        "annoyed",
        "hate",
        "irritated",
        "rage",
        "frustrated",
        "frustrating",
        "outraged",
        "resent",
    ],
    "fear": [
        "fear",
        "afraid",
        "scared",
        "terrified",
        "anxious",
        "nervous",
        "worried",
        "worry",
        "panic",
        "unsafe",
        "frightened",
    ],
    "surprise": [
        "surprised",
        "surprise",
        "shock",
        "shocked",
        "unexpected",
        "astonished",
        "amazed",
        "suddenly",
        "wow",
        "startled",
        "stunned",
    ],
    "trust": [
        "trust",
        "secure",
        "safe",
        "reliable",
        "confident",
        "faith",
        "honest",
        "dependable",
        "assured",
        "believe",
        "support",
        "supported",
    ],
    "anticipation": [
        "anticipate",
        "hope",
        "hopeful",
        "expect",
        "waiting",
        "eager",
        "looking forward",
        "soon",
        "prepare",
        "plan",
        "await",
        "future",
    ],
    "disgust": [
        "disgust",
        "gross",
        "nasty",
        "awful",
        "revolting",
        "sick",
        "horrible",
        "dirty",
        "dislike",
        "repulsive",
        "terrible",
        "worst",
    ],
}


@lru_cache(maxsize=1)
def get_sentiment_analyzer() -> SentimentIntensityAnalyzer:
    """Return a cached VADER sentiment analyzer."""
    return SentimentIntensityAnalyzer()


def analyze_sentiment_vader(text: str) -> tuple[str, float, dict[str, float]]:
    """Analyze sentiment using VADER and return label, confidence, and raw scores."""
    scores = get_sentiment_analyzer().polarity_scores(text or "")
    compound = scores["compound"]

    if compound >= 0.05:
        return "Positive", compound, scores
    if compound <= -0.05:
        return "Negative", abs(compound), scores
    return "Neutral", 1 - abs(compound), scores


def analyze_sentiment_textblob(text: str) -> tuple[str, float]:
    """Analyze sentiment using TextBlob polarity."""
    polarity = TextBlob(text or "").sentiment.polarity

    if polarity > 0:
        return "Positive", polarity
    if polarity < 0:
        return "Negative", abs(polarity)
    return "Neutral", 1.0


def _tokenize(text: str) -> list[str]:
    """Tokenize text for simple fallback lexicon matching."""
    return re.findall(r"[a-zA-Z']+", text.lower())


def _fallback_emotion_scores(text: str) -> dict[str, float]:
    """Return normalized fallback emotion scores."""
    lowered_text = text.lower()
    words = _tokenize(text)
    scores = Counter()

    for emotion_name, keywords in FALLBACK_EMOTION_LEXICON.items():
        for keyword in keywords:
            if " " in keyword:
                scores[emotion_name] += lowered_text.count(keyword)
            else:
                scores[emotion_name] += words.count(keyword)

    total = sum(scores.values())
    if total == 0:
        sentiment, confidence, _ = analyze_sentiment_vader(text)
        if sentiment == "Positive":
            return {"Joy": 0.65, "Trust": 0.2, "Anticipation": 0.15}
        if sentiment == "Negative":
            return {"Sadness": 0.35, "Anger": 0.3, "Fear": 0.2, "Disgust": 0.15}
        return {"Neutral": 1.0}

    return {
        emotion_name.title(): score / total
        for emotion_name, score in scores.items()
        if score > 0
    }


def detect_emotions(text: str) -> dict[str, float]:
    """Detect emotions using NRCLex or a built-in fallback lexicon."""
    cleaned_text = (text or "").strip()
    if not cleaned_text:
        return {"Neutral": 1.0}

    if NRCLEX_AVAILABLE and NRCLex is not None:
        try:
            emotion = NRCLex(cleaned_text)
            emotions = emotion.raw_emotion_scores
            if emotions:
                total = sum(emotions.values())
                nrc_scores = {
                    key.title(): value / total
                    for key, value in emotions.items()
                    if total > 0
                }
                fallback_scores = _fallback_emotion_scores(cleaned_text)
                if fallback_scores != {"Neutral": 1.0}:
                    merged = Counter(nrc_scores)
                    for key, value in fallback_scores.items():
                        merged[key] += value * 0.6
                    merged_total = sum(merged.values())
                    return {key: value / merged_total for key, value in merged.items()}
                return nrc_scores
        except Exception:
            # NRCLex can fail when optional NLP corpora are unavailable.
            # The fallback lexicon keeps the UI working instead of hiding results.
            pass

    return _fallback_emotion_scores(cleaned_text)
