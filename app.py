"""Streamlit application for sentiment, emotion, and emoji analysis."""

from __future__ import annotations

from datetime import datetime

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.database import clear_history, get_history, init_database, save_to_history
from src.emoji_utils import (
    detect_emoji_categories,
    detect_emoji_sentiment,
    detect_emojis,
    score_emoji_sentiment,
)
from src.auth import authenticate_user, create_user, init_auth_tables, list_users
from src.ml_model import model_exists, predict_with_model, train_sentiment_model
from src.sentiment_engine import (
    NRCLEX_AVAILABLE,
    analyze_sentiment_textblob,
    analyze_sentiment_vader,
    detect_emotions,
)
from src.styles import apply_custom_css
from src.visualization import (
    create_confidence_chart,
    create_sentiment_distribution,
    create_wordcloud,
)


st.set_page_config(
    page_title="Sentiment Analysis Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


def initialize_session_state() -> None:
    """Initialize values that must survive Streamlit reruns."""
    defaults = {
        "last_text": "",
        "last_results": None,
        "dataset_result": None,
        "training_result": None,
        "user": None,
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def render_sentiment_card(title: str, sentiment: str, confidence: float) -> None:
    """Render a compact sentiment result card."""
    css_class = {
        "Positive": "positive-sentiment",
        "Negative": "negative-sentiment",
        "Neutral": "neutral-sentiment",
    }.get(sentiment, "neutral-sentiment")

    st.markdown(
        f"""
        <div class="metric-card {css_class}">
            <div class="metric-title">{title}</div>
            <div class="metric-value">{sentiment}</div>
            <div class="metric-subtitle">Confidence: {confidence:.2%}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_vader_gauge(compound_score: float) -> None:
    """Render a VADER compound score gauge."""
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=compound_score,
            domain={"x": [0, 1], "y": [0, 1]},
            title={"text": "Compound Score"},
            gauge={
                "axis": {"range": [-1, 1]},
                "bar": {"color": "#2563eb"},
                "steps": [
                    {"range": [-1, -0.05], "color": "#fecaca"},
                    {"range": [-0.05, 0.05], "color": "#fef3c7"},
                    {"range": [0.05, 1], "color": "#bbf7d0"},
                ],
                "threshold": {
                    "line": {"color": "#111827", "width": 3},
                    "thickness": 0.75,
                    "value": 0,
                },
            },
        )
    )
    fig.update_layout(height=280, margin={"l": 20, "r": 20, "t": 50, "b": 20})
    st.plotly_chart(fig, use_container_width=True)


def render_emotion_analysis(text: str) -> None:
    """Render emotion detection table and chart."""
    st.markdown("### Emotion Analysis")
    emotions = detect_emotions(text)
    emotion_df = (
        pd.DataFrame({"Emotion": list(emotions.keys()), "Score": list(emotions.values())})
        .sort_values("Score", ascending=False)
        .reset_index(drop=True)
    )

    top_emotion = emotion_df.iloc[0]["Emotion"]
    top_score = float(emotion_df.iloc[0]["Score"])
    metric_col1, metric_col2, metric_col3 = st.columns(3)
    metric_col1.metric("Dominant Emotion", top_emotion)
    metric_col2.metric("Emotion Strength", f"{top_score:.2%}")
    metric_col3.metric("Detected Emotions", len(emotion_df))

    chart_df = emotion_df[emotion_df["Emotion"] != "Neutral"].copy()
    if chart_df.empty:
        chart_df = emotion_df.copy()

    table_col, chart_col = st.columns([1, 2])
    with table_col:
        st.dataframe(
            emotion_df.style.format({"Score": "{:.2%}"}),
            use_container_width=True,
            hide_index=True,
        )
        if not NRCLEX_AVAILABLE:
            st.caption("NRCLex is unavailable, so the fallback emotion lexicon is being used.")

    with chart_col:
        fig = px.bar(
            chart_df.sort_values("Score"),
            x="Score",
            y="Emotion",
            orientation="h",
            color="Emotion",
            text=chart_df["Score"].map(lambda value: f"{value:.0%}"),
            title="Emotion Strength",
        )
        fig.update_layout(
            showlegend=False,
            xaxis_tickformat=".0%",
            xaxis_title="Score",
            yaxis_title="",
            height=340,
            margin={"l": 20, "r": 20, "t": 55, "b": 30},
        )
        st.plotly_chart(fig, use_container_width=True)


def render_emoji_analysis(text: str) -> None:
    """Render emoji detection, sentiment, and category analysis."""
    st.markdown("### Emoji Analysis")
    emoji_df, warning = detect_emojis(text)
    emoji_sentiment, positive_count, negative_count = detect_emoji_sentiment(text)
    category_df, top_category = detect_emoji_categories(text)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Emoji Sentiment", emoji_sentiment)
    col2.metric("Positive Emojis", positive_count)
    col3.metric("Negative Emojis", negative_count)
    col4.metric("Top Category", top_category or "None")

    if warning:
        st.warning(warning)

    if emoji_df.empty:
        st.info("No emojis were found in this text.")
        return

    table_col, chart_col = st.columns([1, 2])
    with table_col:
        st.dataframe(emoji_df, use_container_width=True, hide_index=True)

    with chart_col:
        fig = px.bar(
            emoji_df.sort_values("Count"),
            x="Count",
            y="Emoji",
            orientation="h",
            color="Meaning",
            text="Count",
            title="Emoji Frequency",
        )
        fig.update_layout(yaxis_title="", xaxis_title="Count", height=320)
        st.plotly_chart(fig, use_container_width=True)

    if not category_df.empty:
        st.markdown("#### Emoji Categories")
        cat_col1, cat_col2 = st.columns([1, 2])
        with cat_col1:
            st.dataframe(
                category_df.style.format({"Share": "{:.2%}"}),
                use_container_width=True,
                hide_index=True,
            )
        with cat_col2:
            fig = px.bar(
                category_df.sort_values("Count"),
                x="Count",
                y="Category",
                orientation="h",
                color="Category",
                title="Emoji Category Distribution",
                text="Count",
            )
            fig.update_layout(showlegend=False, yaxis_title="", xaxis_title="Count", height=320)
            st.plotly_chart(fig, use_container_width=True)


def combine_text_and_emoji_sentiment(
    vader_sentiment: str,
    vader_confidence: float,
    emoji_sentiment: str,
    positive_emojis: int,
    negative_emojis: int,
    emoji_score: float,
) -> tuple[str, float, str]:
    """Create a final sentiment that accounts for text and emoji signals."""
    text_score = {
        "Positive": vader_confidence,
        "Negative": -vader_confidence,
        "Neutral": 0.0,
    }.get(vader_sentiment, 0.0)

    if positive_emojis + negative_emojis == 0:
        return vader_sentiment, vader_confidence, "No sentiment emojis detected."

    emoji_weight = 0.4 if abs(emoji_score) >= 0.75 else 0.3
    text_weight = 1 - emoji_weight
    combined_score = (text_score * text_weight) + (emoji_score * emoji_weight)

    if combined_score >= 0.05:
        final_sentiment = "Positive"
    elif combined_score <= -0.05:
        final_sentiment = "Negative"
    else:
        final_sentiment = "Neutral"

    confidence = min(abs(combined_score), 1.0)
    explanation = (
        f"Text signal: {vader_sentiment}. Emoji signal: {emoji_sentiment} "
        f"({positive_emojis} positive, {negative_emojis} negative, "
        f"emoji score {emoji_score:+.2f})."
    )
    return final_sentiment, confidence, explanation


def analyze_single_text(text: str) -> dict:
    """Analyze a single text using both supported sentiment engines."""
    vader_sentiment, vader_confidence, vader_scores = analyze_sentiment_vader(text)
    textblob_sentiment, textblob_confidence = analyze_sentiment_textblob(text)
    emoji_sentiment, positive_emojis, negative_emojis = detect_emoji_sentiment(text)
    emoji_score, _, _, emoji_category_counts = score_emoji_sentiment(text)
    final_sentiment, final_confidence, final_reason = combine_text_and_emoji_sentiment(
        vader_sentiment,
        vader_confidence,
        emoji_sentiment,
        positive_emojis,
        negative_emojis,
        emoji_score,
    )
    return {
        "vader_sentiment": vader_sentiment,
        "vader_confidence": vader_confidence,
        "vader_scores": vader_scores,
        "textblob_sentiment": textblob_sentiment,
        "textblob_confidence": textblob_confidence,
        "emoji_sentiment": emoji_sentiment,
        "positive_emojis": positive_emojis,
        "negative_emojis": negative_emojis,
        "emoji_score": emoji_score,
        "emoji_category_counts": emoji_category_counts,
        "final_sentiment": final_sentiment,
        "final_confidence": final_confidence,
        "final_reason": final_reason,
    }


def render_auth_page(conn) -> bool:
    """Render login/signup screens and return authentication status."""
    if st.session_state.user:
        return True

    st.markdown('<h1 class="main-header">Sentiment Analysis Dashboard</h1>', unsafe_allow_html=True)
    login_tab, signup_tab = st.tabs(["Login", "Create Account"])

    with login_tab:
        username = st.text_input("Username", key="login_username")
        password = st.text_input("Password", type="password", key="login_password")
        if st.button("Login", type="primary", use_container_width=True):
            user = authenticate_user(conn, username, password)
            if user:
                st.session_state.user = user
                st.success("Login successful.")
                st.rerun()
            else:
                st.error("Invalid username or password.")
        st.caption("Default admin account: username admin, password admin123")

    with signup_tab:
        new_username = st.text_input("New username", key="signup_username")
        new_password = st.text_input("New password", type="password", key="signup_password")
        confirm_password = st.text_input("Confirm password", type="password", key="signup_confirm")
        if st.button("Create Account", use_container_width=True):
            if new_password != confirm_password:
                st.error("Passwords do not match.")
            else:
                try:
                    create_user(conn, new_username, new_password)
                    st.success("Account created. Please login.")
                except Exception as exc:
                    st.error(f"Could not create account: {exc}")

    return False


def render_text_analysis(conn) -> None:
    """Render the text analysis section."""
    st.markdown("## Text Analysis")
    st.caption("Analyze sentence-level sentiment, emotions, and emoji signals.")

    input_col, guide_col = st.columns([2, 1])
    with input_col:
        text_input = st.text_area(
            "Enter text",
            value=st.session_state.last_text,
            placeholder="Example: I loved the product! The experience was amazing 😀",
            height=150,
        )
        analysis_method = st.selectbox(
            "Analysis method",
            ["VADER Sentiment", "TextBlob", "Both"],
        )
        analyze_button = st.button("Analyze Sentiment", type="primary", use_container_width=True)

    with guide_col:
        st.markdown("### Input Quality")
        st.write("Use complete sentences for stronger sentiment and emotion results.")
        st.write("Emoji analysis works best when the text includes expressive symbols.")

    if analyze_button:
        clean_text = text_input.strip()
        if not clean_text:
            st.warning("Please enter text before running analysis.")
        else:
            with st.spinner("Analyzing text..."):
                results = analyze_single_text(clean_text)
                save_to_history(
                    conn,
                    clean_text,
                    results["vader_sentiment"],
                    results["vader_confidence"],
                    st.session_state.user["id"],
                )
                st.session_state.last_text = clean_text
                st.session_state.last_results = results

    if not st.session_state.last_results:
        return

    results = st.session_state.last_results
    text_for_details = st.session_state.last_text

    st.markdown("---")
    st.markdown("### Sentiment Results")

    final_col, emoji_col = st.columns([1.2, 1])
    with final_col:
        render_sentiment_card(
            "Final Sentiment",
            results["final_sentiment"],
            results["final_confidence"],
        )
        st.caption(results["final_reason"])
    with emoji_col:
        emoji_summary_df = pd.DataFrame(
            [
                {"Signal": "Emoji Sentiment", "Value": results["emoji_sentiment"]},
                {"Signal": "Emoji Score", "Value": f"{results['emoji_score']:+.2f}"},
                {"Signal": "Positive Emojis", "Value": results["positive_emojis"]},
                {"Signal": "Negative Emojis", "Value": results["negative_emojis"]},
                {
                    "Signal": "Emoji Categories",
                    "Value": ", ".join(
                        f"{category}: {count}"
                        for category, count in results["emoji_category_counts"].items()
                    )
                    or "None",
                },
            ]
        )
        st.dataframe(emoji_summary_df, use_container_width=True, hide_index=True)

    if analysis_method in {"VADER Sentiment", "Both"}:
        card_col, json_col, gauge_col = st.columns([1, 1, 1.3])
        with card_col:
            render_sentiment_card(
                "VADER Sentiment",
                results["vader_sentiment"],
                results["vader_confidence"],
            )
        with json_col:
            st.markdown("#### VADER Scores")
            st.json(results["vader_scores"])
        with gauge_col:
            render_vader_gauge(results["vader_scores"]["compound"])

    if analysis_method in {"TextBlob", "Both"}:
        col1, col2 = st.columns([1, 2])
        with col1:
            render_sentiment_card(
                "TextBlob Sentiment",
                results["textblob_sentiment"],
                results["textblob_confidence"],
            )
        with col2:
            comparison_df = pd.DataFrame(
                {
                    "Method": ["VADER", "TextBlob"],
                    "Sentiment": [results["vader_sentiment"], results["textblob_sentiment"]],
                    "Confidence": [results["vader_confidence"], results["textblob_confidence"]],
                }
            )
            st.dataframe(comparison_df, use_container_width=True, hide_index=True)

    st.markdown("---")
    render_emotion_analysis(text_for_details)
    st.markdown("---")
    render_emoji_analysis(text_for_details)


def render_dataset_analysis() -> None:
    """Render CSV upload and bulk sentiment analysis."""
    st.markdown("## Dataset Analysis")
    st.caption("Upload a CSV file, choose a text column, and analyze all rows.")

    uploaded_file = st.file_uploader("Upload CSV file", type="csv")
    if uploaded_file is None:
        return

    try:
        df = pd.read_csv(uploaded_file)
    except Exception as exc:
        st.error(f"Unable to read CSV file: {exc}")
        return

    st.success("Dataset loaded successfully.")
    st.dataframe(df.head(10), use_container_width=True)

    text_columns = df.select_dtypes(include=["object", "string"]).columns.tolist()
    if not text_columns:
        st.error("No text columns found in the uploaded dataset.")
        return

    selected_column = st.selectbox("Select text column", text_columns)

    if st.button("Analyze Dataset", type="primary"):
        with st.spinner("Analyzing dataset rows..."):
            analyzed_df = df.copy()
            sentiments: list[str] = []
            confidences: list[float] = []

            for text in analyzed_df[selected_column].fillna("").astype(str):
                sentiment, confidence, _ = analyze_sentiment_vader(text)
                sentiments.append(sentiment)
                confidences.append(confidence)

            analyzed_df["sentiment"] = sentiments
            analyzed_df["confidence"] = confidences
            st.session_state.dataset_result = (analyzed_df, selected_column)

    if not st.session_state.dataset_result:
        return

    analyzed_df, selected_column = st.session_state.dataset_result
    st.markdown("### Dataset Results")
    sentiment_counts = analyzed_df["sentiment"].value_counts()

    col1, col2, col3, col4 = st.columns(4)
    total_rows = max(len(analyzed_df), 1)
    with col1:
        render_sentiment_card("Positive Rows", "Positive", sentiment_counts.get("Positive", 0) / total_rows)
        st.caption(f"Count: {sentiment_counts.get('Positive', 0)}")
    with col2:
        render_sentiment_card("Negative Rows", "Negative", sentiment_counts.get("Negative", 0) / total_rows)
        st.caption(f"Count: {sentiment_counts.get('Negative', 0)}")
    with col3:
        render_sentiment_card("Neutral Rows", "Neutral", sentiment_counts.get("Neutral", 0) / total_rows)
        st.caption(f"Count: {sentiment_counts.get('Neutral', 0)}")
    with col4:
        st.metric("Total Rows", len(analyzed_df))

    chart_col1, chart_col2 = st.columns(2)
    with chart_col1:
        sentiment_dist = analyzed_df["sentiment"].value_counts().reset_index()
        sentiment_dist.columns = ["sentiment", "count"]
        st.plotly_chart(create_sentiment_distribution(sentiment_dist), use_container_width=True)
    with chart_col2:
        avg_confidence = analyzed_df.groupby("sentiment", as_index=False)["confidence"].mean()
        st.plotly_chart(create_confidence_chart(avg_confidence), use_container_width=True)

    st.markdown("### Word Clouds")
    wc_col1, wc_col2 = st.columns(2)
    with wc_col1:
        positive_texts = analyzed_df.loc[
            analyzed_df["sentiment"] == "Positive", selected_column
        ].astype(str)
        fig = create_wordcloud(positive_texts.tolist(), "Positive")
        if fig:
            st.pyplot(fig)
        else:
            st.info("No positive text available for word cloud.")
    with wc_col2:
        negative_texts = analyzed_df.loc[
            analyzed_df["sentiment"] == "Negative", selected_column
        ].astype(str)
        fig = create_wordcloud(negative_texts.tolist(), "Negative")
        if fig:
            st.pyplot(fig)
        else:
            st.info("No negative text available for word cloud.")

    csv_data = analyzed_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download Analyzed Dataset",
        data=csv_data,
        file_name=f"analyzed_dataset_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv",
    )


def render_model_training() -> None:
    """Render supervised machine learning training and prediction workflow."""
    st.markdown("## ML Model Training")
    st.caption(
        "Upload a labeled CSV dataset, train a TF-IDF + Logistic Regression model, "
        "and use the saved model for future predictions."
    )

    st.markdown("### Dataset Requirements")
    st.write("Your CSV should contain one text column and one label column.")
    st.write("Labels can be Positive, Negative, Neutral, or your own class names.")

    uploaded_file = st.file_uploader("Upload labeled CSV file", type="csv", key="ml_dataset")
    if uploaded_file is None:
        st.info("You can test this section with sample_data/labeled_reviews.csv.")
    else:
        try:
            df = pd.read_csv(uploaded_file)
        except Exception as exc:
            st.error(f"Unable to read CSV file: {exc}")
            return

        st.success("Labeled dataset loaded successfully.")
        st.dataframe(df.head(10), use_container_width=True)

        columns = df.columns.tolist()
        text_column = st.selectbox("Select text column", columns, key="ml_text_column")
        label_column = st.selectbox("Select label column", columns, key="ml_label_column")
        test_size = st.slider("Test data size", min_value=0.1, max_value=0.4, value=0.2, step=0.05)

        if st.button("Train ML Model", type="primary"):
            if text_column == label_column:
                st.error("Text column and label column must be different.")
            else:
                with st.spinner("Training model and calculating evaluation metrics..."):
                    try:
                        st.session_state.training_result = train_sentiment_model(
                            df,
                            text_column,
                            label_column,
                            test_size=test_size,
                        )
                    except Exception as exc:
                        st.error(f"Training failed: {exc}")
                        return

    result = st.session_state.training_result
    if result:
        st.markdown("---")
        st.markdown("### Model Evaluation")

        col1, col2, col3 = st.columns(3)
        col1.metric("Accuracy", f"{result['accuracy']:.2%}")
        col2.metric("Training Rows", result["train_rows"])
        col3.metric("Testing Rows", result["test_rows"])

        chart_col1, chart_col2 = st.columns(2)
        with chart_col1:
            class_df = result["class_distribution"]
            class_df.columns = ["Sentiment", "Count"]
            fig = px.bar(
                class_df,
                x="Sentiment",
                y="Count",
                color="Sentiment",
                title="Class Distribution",
                text="Count",
            )
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

        with chart_col2:
            matrix = result["confusion_matrix"]
            labels = result["labels"]
            fig = px.imshow(
                matrix,
                x=labels,
                y=labels,
                text_auto=True,
                color_continuous_scale="Blues",
                title="Confusion Matrix",
            )
            fig.update_layout(xaxis_title="Predicted", yaxis_title="Actual")
            st.plotly_chart(fig, use_container_width=True)

        report_df = pd.DataFrame(result["report"]).transpose().reset_index()
        report_df = report_df.rename(columns={"index": "Class"})
        st.markdown("### Classification Report")
        st.dataframe(report_df, use_container_width=True, hide_index=True)
        st.success(f"Model saved at: {result['model_path']}")

    st.markdown("---")
    st.markdown("### Predict With Saved ML Model")
    if not model_exists():
        st.warning("No trained model found yet. Train a model first.")
        return

    prediction_text = st.text_area(
        "Enter text for ML model prediction",
        placeholder="Example: The product quality is excellent and support was helpful.",
        key="ml_prediction_text",
    )
    if st.button("Predict With ML Model"):
        if not prediction_text.strip():
            st.warning("Please enter text for prediction.")
        else:
            try:
                prediction, confidence = predict_with_model(prediction_text.strip())
            except Exception as exc:
                st.error(f"Prediction failed: {exc}")
                return

            if confidence is None:
                st.metric("Predicted Sentiment", prediction)
            else:
                st.metric("Predicted Sentiment", prediction, f"{confidence:.2%} confidence")


def render_history(conn) -> None:
    """Render analysis history with filters and export controls."""
    st.markdown("## Analysis History")
    st.caption("Review saved single-text analysis records from local SQLite storage.")

    user = st.session_state.user
    records = get_history(
        conn,
        limit=500,
        user_id=user["id"],
        include_all=user["role"] == "admin",
    )
    if not records:
        st.info("No history found. Analyze text first to create history records.")
        return

    history_df = pd.DataFrame(records, columns=["Text", "Sentiment", "Confidence", "Timestamp"])
    history_df["Timestamp"] = pd.to_datetime(history_df["Timestamp"])

    col1, col2, col3 = st.columns(3)
    sentiment_filter = col1.selectbox("Sentiment", ["All", "Positive", "Negative", "Neutral"])
    use_date_filter = col2.checkbox("Filter by date")
    selected_date = col2.date_input("Date") if use_date_filter else None
    search_term = col3.text_input("Search text")

    if sentiment_filter != "All":
        history_df = history_df[history_df["Sentiment"] == sentiment_filter]
    if selected_date:
        history_df = history_df[history_df["Timestamp"].dt.date == selected_date]
    if search_term:
        history_df = history_df[
            history_df["Text"].str.contains(search_term, case=False, na=False, regex=False)
        ]

    stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)
    stat_col1.metric("Total Analyses", len(history_df))
    stat_col2.metric("Positive", int((history_df["Sentiment"] == "Positive").sum()))
    stat_col3.metric("Negative", int((history_df["Sentiment"] == "Negative").sum()))
    stat_col4.metric("Neutral", int((history_df["Sentiment"] == "Neutral").sum()))

    st.markdown("### Records")
    items_per_page = 10
    total_pages = max(1, (len(history_df) - 1) // items_per_page + 1)
    page = st.number_input("Page", min_value=1, max_value=total_pages, value=1)
    start_idx = (page - 1) * items_per_page
    page_data = history_df.iloc[start_idx : start_idx + items_per_page]

    for _, row in page_data.iterrows():
        sentiment_class = f"{row['Sentiment'].lower()}-sentiment"
        st.markdown(
            f"""
            <div class="history-card {sentiment_class}">
                <strong>{row['Sentiment']}</strong>
                <span>Confidence: {row['Confidence']:.2%}</span>
                <p>{row['Text']}</p>
                <small>{row['Timestamp'].strftime('%Y-%m-%d %H:%M:%S')}</small>
            </div>
            """,
            unsafe_allow_html=True,
        )

    action_col1, action_col2 = st.columns(2)
    with action_col1:
        csv_data = history_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Export History",
            data=csv_data,
            file_name=f"sentiment_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
        )
    with action_col2:
        if st.button("Clear All History"):
            clear_history(conn)
            st.success("History cleared.")
            st.rerun()


def render_admin_panel(conn) -> None:
    """Render admin-only user and system overview."""
    st.markdown("## Admin Panel")
    if st.session_state.user["role"] != "admin":
        st.error("Admin access only.")
        return

    users_df = pd.DataFrame(
        list_users(conn),
        columns=["User ID", "Username", "Role", "Created At", "Analyses"],
    )
    st.markdown("### Registered Users")
    st.dataframe(users_df, use_container_width=True, hide_index=True)

    st.markdown("### System Files")
    st.write("Model artifact:", "Available" if model_exists() else "Not trained yet")
    st.write("Sample dataset:", "sample_data/labeled_reviews.csv")
    st.write("Project report:", "docs/PROJECT_REPORT.md")


def main() -> None:
    """Application entry point."""
    apply_custom_css()
    initialize_session_state()
    conn = init_database()
    init_auth_tables(conn)

    if not render_auth_page(conn):
        conn.close()
        return

    st.markdown('<h1 class="main-header">Sentiment Analysis Dashboard</h1>', unsafe_allow_html=True)

    with st.sidebar:
        st.markdown(f"**User:** {st.session_state.user['username']}")
        st.caption(f"Role: {st.session_state.user['role']}")
        if st.button("Logout"):
            st.session_state.user = None
            st.rerun()
        st.markdown("---")
        st.markdown("# Navigation")
        sections = ["Text Analysis", "Dataset Analysis", "ML Model Training", "History"]
        if st.session_state.user["role"] == "admin":
            sections.append("Admin Panel")
        section = st.radio(
            "Choose section",
            sections,
        )
        st.markdown("---")
        st.markdown("### About")
        st.info(
            "A final-year project dashboard for sentiment, emotion, emoji, and dataset analysis."
        )

    try:
        if section == "Text Analysis":
            render_text_analysis(conn)
        elif section == "Dataset Analysis":
            render_dataset_analysis()
        elif section == "ML Model Training":
            render_model_training()
        elif section == "Admin Panel":
            render_admin_panel(conn)
        else:
            render_history(conn)
    finally:
        conn.close()


if __name__ == "__main__":
    main()
