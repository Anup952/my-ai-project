"""Chart and word cloud helpers."""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
from wordcloud import WordCloud


SENTIMENT_COLORS = {
    "Positive": "#16a34a",
    "Negative": "#dc2626",
    "Neutral": "#d97706",
}


def create_sentiment_distribution(df: pd.DataFrame):
    """Create a pie chart for sentiment count distribution."""
    return px.pie(
        df,
        names="sentiment",
        values="count",
        title="Sentiment Distribution",
        color="sentiment",
        color_discrete_map=SENTIMENT_COLORS,
        hole=0.35,
    )


def create_confidence_chart(df: pd.DataFrame, title: str = "Average Confidence by Sentiment"):
    """Create a bar chart for confidence-like values."""
    fig = px.bar(
        df,
        x="sentiment",
        y="confidence",
        title=title,
        color="sentiment",
        text_auto=".2f",
        color_discrete_map=SENTIMENT_COLORS,
    )
    fig.update_layout(showlegend=False, xaxis_title="", yaxis_title="")
    return fig


def create_wordcloud(text_list: list[str], sentiment_type: str):
    """Create a Matplotlib word cloud figure from a list of texts."""
    text = " ".join(text_list).strip()
    if not text:
        return None

    colormap = "Greens" if sentiment_type == "Positive" else "Reds"
    wordcloud = WordCloud(
        width=900,
        height=450,
        background_color="white",
        colormap=colormap,
        collocations=False,
    ).generate(text)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation="bilinear")
    ax.axis("off")
    ax.set_title(f"{sentiment_type} Word Cloud", fontsize=16, fontweight="bold")
    return fig

