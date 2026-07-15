# Emotion-Aware Learning Assistant — Prototype

A working local prototype of the emotion-aware learning support platform.
Run it, poke at it, and use it as the scaffold for the full production build
described in the original spec.

## Quick start

```bash
pip install -r requirements.txt
streamlit run app.py
```

Open the URL Streamlit prints (usually http://localhost:8501).

## What's real vs. simulated in this prototype

| Spec component | This prototype | Why |
|---|---|---|
| Emotion classification | Two real, independently trained classifiers (TF-IDF+LogisticRegression, TF-IDF+MLP) on a bundled 320-example synthetic dataset covering **16 emotions**: Bored, Confident, Confused, Curious, Frustrated, Anxious, Excited, Overwhelmed, Motivated, Disappointed, Proud, Relieved, Discouraged, Determined, Satisfied, Embarrassed | No GPU / internet access to huggingface.co or Kaggle in this environment; the architecture (dual model, confidence comparison, 0.30 mixed-emotion threshold) is fully implemented and swappable |
| Response generation | Real `google-generativeai` (Gemini) call if you provide an API key in the sidebar, otherwise a structured template fallback (still enforces Validation → Hints → Closure) | Works offline out of the box; upgrades automatically once you add a key |
| Persistence | Local SQLite (`telemetry.db`) | No BigQuery project/credentials available here; same event schema, drop-in swappable |
| Deployment | Dockerfile for Cloud Run included and follows the spec's tuning guidance | You'll need your own GCP project to actually deploy |

## Upgrading to the full production spec

**1. Real BiLSTM / BERT models**
- Train a Keras BiLSTM on GoEmotions (mapped to the 5 labels) on Kaggle GPU, export as `.h5`/`SavedModel`.
- Fine-tune a `bert-base-uncased` (HuggingFace `transformers`) on the same data.
- In `emotion_model.py`, replace `_train_fast_model` / `_train_deep_model` and `_Pipeline._score` with calls to your loaded models. Keep the return contract (`{label: probability}` dict) the same and everything else — thresholding, the UI, the dashboard — keeps working unchanged.

**2. Gemini API**
- Get a key from Google AI Studio, set `GEMINI_API_KEY` as an env var or paste it into the sidebar.
- `prompt_engine.py` uses the current `google-genai` SDK (`from google import genai`) against the stable `gemini-2.5-flash` model — the previous `google-generativeai` package and `gemini-2.0-flash` model have both been shut down/deprecated by Google.

**3. BigQuery instead of SQLite**
- Replace the body of `log_event` / `fetch_history` in `telemetry.py` with `google-cloud-bigquery` client calls (`client.insert_rows_json`, `client.query`). Keep the same field names to avoid touching `app.py`.

**4. Deploy to Cloud Run**
```bash
gcloud builds submit --tag gcr.io/YOUR_PROJECT/emotion-learning-app
gcloud run deploy emotion-learning-app \
  --image gcr.io/YOUR_PROJECT/emotion-learning-app \
  --memory 4Gi \
  --min-instances 1 \
  --allow-unauthenticated
```
(Bump `--memory` to `8Gi` once you load real BERT weights.)

## File map

```
app.py                 # Streamlit UI (3 tabs), Module 4 feature requirements
emotion_model.py        # Module 2: dual-model pipeline + mixed-emotion thresholding
prompt_engine.py         # Module 3: Gemini call + template fallback, 3-pillar system prompt
telemetry.py             # Local persistence (swap for BigQuery)
data/emotion_dataset.py  # Bundled synthetic training data (swap for GoEmotions/ISEAR)
requirements.txt
Dockerfile               # Module 4: Cloud Run deployment
```
