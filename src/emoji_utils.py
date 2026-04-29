"""Emoji detection and emoji sentiment helpers."""

from __future__ import annotations

from collections import Counter

import pandas as pd

try:
    import emoji
except ImportError:  # pragma: no cover - handled gracefully at runtime
    emoji = None


EMOJI_CATEGORY_MAP = {
    "Joy": [
        "\U0001f600",
        "\U0001f603",
        "\U0001f604",
        "\U0001f601",
        "\U0001f606",
        "\U0001f60a",
        "\U0001f642",
        "\U0001f638",
        "\U0001f973",
        "\U0001f929",
        "\U0001f602",
        "\U0001f923",
        "\u2728",
    ],
    "Love": [
        "\U0001f60d",
        "\U0001f970",
        "\U0001f618",
        "\u2764\ufe0f",
        "\u2764",
        "\U0001f496",
        "\U0001f498",
        "\U0001f49d",
        "\U0001f495",
        "\U0001f49e",
        "\U0001f493",
    ],
    "Anger": ["\U0001f621", "\U0001f620", "\U0001f92c", "\U0001f624", "\U0001f47f"],
    "Sadness": [
        "\U0001f622",
        "\U0001f62d",
        "\U0001f61e",
        "\U0001f614",
        "\U0001f61f",
        "\U0001f494",
        "\U0001f940",
    ],
    "Fear": ["\U0001f628", "\U0001f630", "\U0001f631", "\U0001fae3", "\U0001f627", "\U0001f626"],
    "Surprise": ["\U0001f62e", "\U0001f632", "\U0001f62f", "\U0001f633", "\U0001f92f", "\U0001fae2"],
    "Approval": ["\U0001f44d", "\U0001f44f", "\U0001f64c", "\U0001f4aa", "\U0001f4af", "\U0001f525"],
    "Disapproval": ["\U0001f44e", "\U0001f644", "\U0001f612", "\U0001f610", "\U0001f928"],
}

EMOJI_SENTIMENT_MAP = {
    "Positive": (
        EMOJI_CATEGORY_MAP["Joy"]
        + EMOJI_CATEGORY_MAP["Love"]
        + EMOJI_CATEGORY_MAP["Approval"]
    ),
    "Negative": (
        EMOJI_CATEGORY_MAP["Anger"]
        + EMOJI_CATEGORY_MAP["Sadness"]
        + EMOJI_CATEGORY_MAP["Fear"]
        + EMOJI_CATEGORY_MAP["Disapproval"]
    ),
}

EMOJI_CATEGORY_SENTIMENT = {
    "Joy": 1.0,
    "Love": 1.0,
    "Approval": 0.8,
    "Surprise": 0.15,
    "Anger": -1.0,
    "Sadness": -0.9,
    "Fear": -0.75,
    "Disapproval": -0.8,
}


def normalize_emoji_text(text: str) -> str:
    """Remove variation selectors that can prevent direct emoji matching."""
    return (text or "").replace("\ufe0f", "")


def _detected_symbols(text: str) -> list[str]:
    """Return emoji symbols found in text with normalized variants."""
    normalized_text = normalize_emoji_text(text)
    if emoji is None:
        return list(normalized_text)

    detected = [item["emoji"] for item in emoji.emoji_list(text)]
    detected.extend(normalize_emoji_text(symbol) for symbol in list(detected))
    for symbol_list in EMOJI_CATEGORY_MAP.values():
        for symbol in symbol_list:
            normalized_symbol = normalize_emoji_text(symbol)
            if symbol in text or normalized_symbol in normalized_text:
                detected.append(symbol)
                detected.append(normalized_symbol)
    return detected


def detect_emojis(text: str) -> tuple[pd.DataFrame, str | None]:
    """Return emoji frequency records and an optional dependency warning."""
    text = text or ""
    warning = None

    if emoji is None:
        warning = "Install emoji for full emoji detection: pip install emoji"
        found_symbols = []
        for symbol_list in EMOJI_CATEGORY_MAP.values():
            found_symbols.extend([symbol for symbol in symbol_list if symbol in text])
        counts = Counter(found_symbols)
        rows = [
            {"Emoji": symbol, "Meaning": "Known sentiment emoji", "Count": count}
            for symbol, count in counts.items()
        ]
        return pd.DataFrame(rows, columns=["Emoji", "Meaning", "Count"]), warning

    found_emojis = emoji.emoji_list(text)
    if not found_emojis:
        fallback_symbols = [
            symbol
            for symbol_list in EMOJI_CATEGORY_MAP.values()
            for symbol in symbol_list
            if symbol in text or normalize_emoji_text(symbol) in normalize_emoji_text(text)
        ]
        counts = Counter(fallback_symbols)
        rows = [
            {"Emoji": symbol, "Meaning": "Known sentiment emoji", "Count": count}
            for symbol, count in counts.items()
        ]
        return pd.DataFrame(rows, columns=["Emoji", "Meaning", "Count"]), None

    counts: dict[str, dict[str, int | str]] = {}
    for item in found_emojis:
        symbol = item["emoji"]
        meaning = emoji.demojize(symbol).replace(":", "").replace("_", " ").title()
        counts.setdefault(symbol, {"Emoji": symbol, "Meaning": meaning, "Count": 0})
        counts[symbol]["Count"] += 1

    return pd.DataFrame(counts.values()), warning


def detect_emoji_sentiment(text: str) -> tuple[str, int, int]:
    """Classify emoji sentiment as positive, negative, mixed, or neutral."""
    _, positive_count, negative_count, _ = score_emoji_sentiment(text)

    if positive_count > negative_count:
        sentiment = "Positive"
    elif negative_count > positive_count:
        sentiment = "Negative"
    elif positive_count == 0 and negative_count == 0:
        sentiment = "Neutral"
    else:
        sentiment = "Mixed"

    return sentiment, positive_count, negative_count


def score_emoji_sentiment(text: str) -> tuple[float, int, int, dict[str, int]]:
    """Return numeric emoji score, positive count, negative count, and category counts."""
    category_df, _ = detect_emoji_categories(text)
    if category_df.empty:
        return 0.0, 0, 0, {}

    category_counts = {
        str(row["Category"]): int(row["Count"])
        for _, row in category_df.iterrows()
    }
    weighted_score = 0.0
    positive_count = 0
    negative_count = 0
    total = sum(category_counts.values())

    for category, count in category_counts.items():
        weight = EMOJI_CATEGORY_SENTIMENT.get(category, 0.0)
        weighted_score += weight * count
        if weight > 0:
            positive_count += count
        elif weight < 0:
            negative_count += count

    if total == 0:
        return 0.0, 0, 0, category_counts

    return weighted_score / total, positive_count, negative_count, category_counts


def detect_emoji_categories(text: str) -> tuple[pd.DataFrame, str | None]:
    """Return category counts for recognized emojis."""
    category_counts = Counter()
    emoji_df, _ = detect_emojis(text)

    if not emoji_df.empty:
        for _, row in emoji_df.iterrows():
            detected_symbol = str(row["Emoji"])
            normalized_detected = normalize_emoji_text(detected_symbol)
            count = int(row["Count"])
            for category, symbols in EMOJI_CATEGORY_MAP.items():
                normalized_symbols = {normalize_emoji_text(symbol) for symbol in symbols}
                if detected_symbol in symbols or normalized_detected in normalized_symbols:
                    category_counts[category] += count
                    break

    if not category_counts:
        return pd.DataFrame(columns=["Category", "Count", "Share"]), None

    total = sum(category_counts.values())
    rows = [
        {"Category": category, "Count": count, "Share": count / total}
        for category, count in category_counts.most_common()
    ]
    return pd.DataFrame(rows), category_counts.most_common(1)[0][0]
