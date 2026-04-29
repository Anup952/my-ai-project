# Sentiment Analysis Dashboard - Project Report

## Abstract

This project is a web-based sentiment analysis dashboard developed using Python and Streamlit. It analyzes user text and CSV datasets to classify sentiment as positive, negative, or neutral. The system also includes emotion detection, emoji analysis, visual analytics, local history storage, and a supervised machine learning training module.

## Problem Statement

Organizations receive large volumes of feedback through reviews, comments, surveys, and social media. Manually reading this text is time-consuming. The objective of this project is to automate sentiment extraction and provide interactive visual insights from text data.

## Objectives

- Build a user-friendly dashboard for text sentiment analysis.
- Support both rule-based and supervised machine learning methods.
- Analyze emotions and emojis for deeper understanding.
- Allow CSV upload for bulk sentiment analysis.
- Store previous analyses using SQLite.
- Provide login, signup, logout, and admin panel features.
- Provide charts, word clouds, and downloadable results.

## Methodology

The project uses VADER and TextBlob for immediate sentiment analysis. For supervised learning, it uses TF-IDF vectorization and Logistic Regression. The user can upload a labeled dataset, train the model, view evaluation metrics, and save the trained model for later prediction.

## Modules

- Text Analysis: Performs VADER, TextBlob, emotion, and emoji analysis.
- Dataset Analysis: Processes uploaded CSV files and visualizes sentiment distribution.
- ML Training: Trains and evaluates a supervised sentiment classifier.
- History: Stores and displays previous text analysis results.
- Authentication: Manages user registration, login sessions, and admin access.
- Visualization: Generates pie charts, bar charts, confusion matrix, and word clouds.

## Technologies Used

- Python
- Streamlit
- Pandas
- Scikit-learn
- VADER Sentiment
- TextBlob
- Plotly
- Matplotlib
- WordCloud
- SQLite

## Future Scope

- Add multilingual sentiment analysis.
- Integrate transformer models such as BERT.
- Add user authentication and admin dashboard.
- Deploy the project on Streamlit Community Cloud or Render.
- Connect to live social media APIs.
