"""Centralized Streamlit CSS styling."""

from __future__ import annotations

import streamlit as st


def apply_custom_css() -> None:
    """Apply the dashboard visual theme."""
    st.markdown(
        """
        <style>
        [data-testid="stAppViewContainer"] {
            background: #f8fafc;
            color: #111827;
        }

        section[data-testid="stSidebar"] {
            background: #0f172a;
        }

        section[data-testid="stSidebar"] * {
            color: #f8fafc;
        }

        .main-header {
            font-size: 2.6rem;
            font-weight: 800;
            text-align: center;
            margin: 0.5rem 0 1.75rem;
            color: #111827;
            letter-spacing: 0;
        }

        .metric-card {
            border-radius: 8px;
            padding: 1rem;
            border: 1px solid #e5e7eb;
            background: #ffffff;
            box-shadow: 0 8px 22px rgba(15, 23, 42, 0.08);
            min-height: 118px;
        }

        .metric-title {
            font-size: 0.9rem;
            font-weight: 700;
            color: #374151;
            margin-bottom: 0.4rem;
        }

        .metric-value {
            font-size: 1.45rem;
            font-weight: 800;
            color: #111827;
        }

        .metric-subtitle {
            margin-top: 0.35rem;
            color: #4b5563;
            font-weight: 600;
        }

        .positive-sentiment {
            background: #ecfdf5;
            border-color: #86efac;
        }

        .negative-sentiment {
            background: #fef2f2;
            border-color: #fca5a5;
        }

        .neutral-sentiment {
            background: #fffbeb;
            border-color: #fde68a;
        }

        .history-card {
            border-radius: 8px;
            padding: 0.9rem 1rem;
            margin: 0.65rem 0;
            border: 1px solid #e5e7eb;
        }

        .history-card span {
            margin-left: 0.75rem;
            color: #4b5563;
        }

        .history-card p {
            margin: 0.5rem 0;
            color: #1f2937;
        }

        .history-card small {
            color: #6b7280;
        }

        .stButton > button {
            border-radius: 8px;
            font-weight: 700;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

