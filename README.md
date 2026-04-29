# Sentiment Analysis Dashboard

A production-ready Streamlit project for sentiment, emotion, and emoji analysis. The application supports single text analysis, bulk CSV analysis, visualization, local history storage, and exportable results.

## Features

- Text sentiment analysis using VADER and TextBlob
- Login, signup, logout, and role-based admin panel
- Emotion detection using NRCLex when available, with a built-in fallback lexicon
- Emoji frequency, category, and emoji-based sentiment detection
- Bulk CSV sentiment analysis
- Supervised ML model training with TF-IDF and Logistic Regression
- Accuracy, confusion matrix, and classification report
- Saved model prediction using Joblib
- Interactive charts using Plotly
- Word cloud generation for positive and negative text groups
- SQLite-based analysis history
- Export analyzed datasets and history as CSV files
- Clean modular Python code suitable for academic review and extension

## Project Structure

```text
sentiment-analysis-dashboard/
├── app.py
├── requirements.txt
├── README.md
├── docs/
│   └── PROJECT_REPORT.md
├── sample_data/
│   ├── labeled_reviews.csv
│   └── reviews.csv
└── src/
    ├── __init__.py
    ├── auth.py
    ├── database.py
    ├── emoji_utils.py
    ├── ml_model.py
    ├── sentiment_engine.py
    ├── styles.py
    └── visualization.py
```

## Installation

Create and activate a virtual environment:

```bash
python -m venv .venv
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
python -m textblob.download_corpora
```

Run the app:

```bash
streamlit run app.py
```

## Login Details

The app creates one default admin account automatically:

```text
Username: admin
Password: admin123
```

Regular users can create accounts from the signup tab. Admin users can view the Admin Panel.

If you already created a virtual environment named `venv`, activate it with:

```bash
venv\Scripts\activate
```

## Dataset Format

Upload a CSV file with at least one text column. Example:

```csv
review
"I loved the product, it was amazing!"
"The service was slow and disappointing."
```

The app lets you choose which text column to analyze.

## ML Training Dataset Format

For the ML Model Training section, upload a CSV with one text column and one sentiment label column:

```csv
review,sentiment
"I loved the product",Positive
"The delivery was terrible",Negative
"It was okay",Neutral
```

You can test this feature with:

```text
sample_data/labeled_reviews.csv
```

After training, the model is saved at:

```text
models/sentiment_model.joblib
```

## Academic Documentation

A ready project report draft is available at:

```text
docs/PROJECT_REPORT.md
```

## Notes

- `NRCLex` is optional. If it fails to install or is unavailable, the app automatically uses a fallback emotion lexicon.
- History is stored locally in `sentiment_history.db`.
- Change the default admin password before using this outside a college demo environment.
- This project is designed for educational and demonstration purposes. For very large datasets, consider batch processing outside Streamlit.
