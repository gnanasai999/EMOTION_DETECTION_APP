"""
AI-Driven Emotion Detection & Personalized Learning Support Platform.
Prototype entry point -- run with: streamlit run app.py
"""

import json
import os

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from emotion_model import get_pipeline, LABELS
from prompt_engine import generate_response
from telemetry import log_event, fetch_history, clear_history
from zero_shot_model import get_zero_shot_pipeline

st.set_page_config(
    page_title="Emotion-Aware Learning Assistant",
    page_icon="🎓",
    layout="wide",
)

pipeline = get_pipeline()

# ---------------------------------------------------------------------------
# Sidebar: config
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### ⚙️ Configuration")
    api_key_input = st.text_input(
        "Gemini API key (optional)",
        type="password",
        value=os.environ.get("GEMINI_API_KEY", ""),
        help="Leave blank to use the built-in template-based response generator.",
    )
    st.caption(
        "Without a key, responses are generated locally using rule-based "
        "templates that still follow the Validation → Hints → Closure structure."
    )
    st.divider()
    st.markdown("### 🗑️ Data")
    if st.button("Clear logged history"):
        clear_history()
        st.success("History cleared.")

st.title("🎓 Emotion-Aware Learning Assistant")
st.caption(
    "Prototype build — dual-model emotion detection + empathetic AI guidance. "
    "See README.md for how to move this to Cloud Run / BigQuery / a fine-tuned BERT."
)

tab1, tab2, tab3 = st.tabs(
    ["💬 Learning Assistant", "📊 Analytics & Trends", "🛠️ Model Comparison & Dev Tools"]
)

# ---------------------------------------------------------------------------
# TAB 1: Learning Assistant Interface
# ---------------------------------------------------------------------------
with tab1:
    st.subheader("What's your study challenge right now?")
    user_text = st.text_area(
        "Describe what you're working on and how it's going...",
        placeholder="e.g. I'm completely lost on recursion, I've read the chapter three times and it still doesn't click.",
        height=120,
    )
    go_btn = st.button("Analyze & Get Guidance", type="primary")

    if go_btn and user_text.strip():
        with st.spinner("Analyzing emotional state..."):
            fast, deep, ensemble = pipeline.predict_all(user_text)

        st.markdown("#### Detected emotional state")
        badge_cols = st.columns(len(ensemble.mixed_labels))
        for col, label in zip(badge_cols, ensemble.mixed_labels):
            col.metric(label, f"{ensemble.scores[label]*100:.0f}%")

        score_col, chart_col = st.columns([1, 1])
        with score_col:
            st.markdown("**Fast model vs Deep model confidence**")
            comp_df = pd.DataFrame(
                {
                    "Emotion": LABELS,
                    "Fast Model": [fast.scores[l] for l in LABELS],
                    "Deep Model": [deep.scores[l] for l in LABELS],
                }
            )
            st.dataframe(comp_df.set_index("Emotion"), use_container_width=True)
        with chart_col:
            fig = go.Figure()
            fig.add_trace(go.Bar(name="Fast Model", x=LABELS, y=[fast.scores[l] for l in LABELS]))
            fig.add_trace(go.Bar(name="Deep Model", x=LABELS, y=[deep.scores[l] for l in LABELS]))
            fig.add_hline(y=0.30, line_dash="dot", annotation_text="mixed-emotion threshold (0.30)")
            fig.update_layout(barmode="group", height=300, margin=dict(t=10, b=10))
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("#### 🤖 Personalized guidance")
        with st.spinner("Generating response..."):
            response_text, source = generate_response(
                user_text, ensemble.mixed_labels, api_key=api_key_input or None
            )
        st.info(response_text)
        st.caption(f"Response source: `{source}`")

        log_event(user_text, fast, deep, ensemble, source)

    elif go_btn:
        st.warning("Please enter some text describing your study challenge first.")

# ---------------------------------------------------------------------------
# TAB 2: Analytics & Trends Dashboard
# ---------------------------------------------------------------------------
with tab2:
    st.subheader("Historic emotion trends")
    df = fetch_history()

    if df.empty:
        st.info("No sessions logged yet — use the Learning Assistant tab to generate data.")
    else:
        df["ensemble_scores_dict"] = df["ensemble_scores"].apply(json.loads)
        expanded = pd.json_normalize(df["ensemble_scores_dict"])
        expanded["ts"] = pd.to_datetime(df["ts"], unit="s")
        expanded["session"] = range(1, len(expanded) + 1)

        st.markdown("**Mixed-emotion distribution over sessions**")
        long_df = expanded.melt(
            id_vars=["session", "ts"], value_vars=LABELS, var_name="Emotion", value_name="Score"
        )
        fig1 = px.area(
            long_df, x="session", y="Score", color="Emotion", groupnorm=None,
            title="Emotion score composition per session",
        )
        st.plotly_chart(fig1, use_container_width=True)

        c1, c2 = st.columns(2)
        with c1:
            avg_confidence = expanded["Confident"].mean()
            avg_frustration = expanded["Frustrated"].mean()
            st.metric("Avg. Confidence level", f"{avg_confidence*100:.1f}%")
            st.metric("Avg. Frustration level", f"{avg_frustration*100:.1f}%")
        with c2:
            fig2 = px.line(
                expanded, x="session", y=["Confident", "Frustrated"],
                title="Confidence vs Frustration over time",
            )
            st.plotly_chart(fig2, use_container_width=True)

        st.markdown("**Top emotion per session (log)**")
        st.dataframe(
            df[["ts", "user_text", "ensemble_top", "mixed_labels", "response_source"]]
            .rename(columns={"ts": "unix_ts"}),
            use_container_width=True,
        )

# ---------------------------------------------------------------------------
# TAB 3: Model Comparison & Dev Tools
# ---------------------------------------------------------------------------
with tab3:
    st.subheader("Latency & system performance")
    df = fetch_history()

    if df.empty:
        st.info("No sessions logged yet — use the Learning Assistant tab to generate data.")
    else:
        c1, c2, c3 = st.columns(3)
        c1.metric("Avg Fast Model latency", f"{df['fast_latency_ms'].mean():.2f} ms")
        c2.metric("Avg Deep Model latency", f"{df['deep_latency_ms'].mean():.2f} ms")
        c3.metric("Sessions logged", f"{len(df)}")

        fig = go.Figure()
        fig.add_trace(go.Scatter(y=df["fast_latency_ms"], name="Fast Model", mode="lines+markers"))
        fig.add_trace(go.Scatter(y=df["deep_latency_ms"], name="Deep Model", mode="lines+markers"))
        fig.update_layout(title="Inference latency per request (ms)", height=350)
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("**Agreement between models**")
        agree = (df["fast_top"] == df["deep_top"]).mean()
        st.metric("Fast/Deep top-label agreement rate", f"{agree*100:.1f}%")

    st.divider()
    st.subheader("🧪 Zero-Shot Classifier (Strategy 3 — no training required)")
    st.caption(
        "Scores arbitrary text against all 16 labels using a HuggingFace NLI "
        "zero-shot pipeline (`facebook/bart-large-mnli`), independent of the "
        "fast/deep models trained on the bundled dataset above."
    )
    zs_text = st.text_area(
        "Text to classify",
        placeholder="e.g. I finally understand recursion, this is such a relief.",
        key="zero_shot_input",
        height=90,
    )
    zs_btn = st.button("Run zero-shot classification")

    if zs_btn:
        if not zs_text.strip():
            st.warning("Enter some text first.")
        else:
            zsp = get_zero_shot_pipeline()
            with st.spinner("Loading zero-shot model (first run downloads ~1.6GB)..."):
                available = zsp.is_available()

            if not available:
                st.error(
                    "Zero-shot classifier unavailable: "
                    f"`{zsp.last_error()}`\n\n"
                    "This needs `transformers` + `torch` installed and network access "
                    "to huggingface.co on first run to download `facebook/bart-large-mnli`. "
                    "Add `transformers` and `torch` to requirements.txt and rerun in an "
                    "environment with internet access."
                )
            else:
                with st.spinner("Scoring against all 16 labels..."):
                    zs_result = zsp.predict(zs_text)
                    fast_zs = pipeline.predict(zs_text, "fast")
                    deep_zs = pipeline.predict(zs_text, "deep")

                st.markdown("**Detected labels (above 0.30 threshold)**")
                badge_cols = st.columns(len(zs_result.mixed_labels))
                for col, label in zip(badge_cols, zs_result.mixed_labels):
                    col.metric(label, f"{zs_result.scores[label]*100:.0f}%")

                zs_comp_df = pd.DataFrame(
                    {
                        "Emotion": LABELS,
                        "Fast Model": [fast_zs.scores[l] for l in LABELS],
                        "Deep Model": [deep_zs.scores[l] for l in LABELS],
                        "Zero-Shot": [zs_result.scores[l] for l in LABELS],
                    }
                )
                fig_zs = go.Figure()
                fig_zs.add_trace(go.Bar(name="Fast Model", x=LABELS, y=zs_comp_df["Fast Model"]))
                fig_zs.add_trace(go.Bar(name="Deep Model", x=LABELS, y=zs_comp_df["Deep Model"]))
                fig_zs.add_trace(go.Bar(name="Zero-Shot", x=LABELS, y=zs_comp_df["Zero-Shot"]))
                fig_zs.add_hline(y=0.30, line_dash="dot", annotation_text="mixed-emotion threshold (0.30)")
                fig_zs.update_layout(barmode="group", height=350, margin=dict(t=10, b=10))
                st.plotly_chart(fig_zs, use_container_width=True)

                lat1, lat2, lat3 = st.columns(3)
                lat1.metric("Fast Model latency", f"{fast_zs.latency_ms:.2f} ms")
                lat2.metric("Deep Model latency", f"{deep_zs.latency_ms:.2f} ms")
                lat3.metric("Zero-Shot latency", f"{zs_result.latency_ms:.2f} ms")

    st.divider()
    st.markdown("**System parameters**")
    st.json(
        {
            "mixed_emotion_threshold": 0.30,
            "fast_model": "TF-IDF (1-2 gram) + Logistic Regression",
            "deep_model": "TF-IDF (1-2 gram) + Calibrated MLP (64,32)",
            "training_examples": 200,
            "labels": LABELS,
            "persistence": "SQLite (telemetry.db) — swap for BigQuery in prod",
            "response_generator": "Gemini API (google-generativeai) with template fallback",
        }
    )
