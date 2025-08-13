import io
import re
import pandas as pd
import plotly.express as px
import streamlit as st
from transformers import pipeline

st.set_page_config(
    page_title="Analyse de Sentiments",
    page_icon="🧠",
    layout="wide",
)

st.title("Analyse de Sentiments – Dashboard")

MODEL_DISPLAY_NAMES = {
    "cardiffnlp/twitter-xlm-roberta-base-sentiment": "Multilingue (Roberta)",
    "distilbert-base-uncased-finetuned-sst-2-english": "Anglais (DistilBERT)"
}

@st.cache_resource(show_spinner=True)
def load_pipeline(model_name: str):
    return pipeline("sentiment-analysis", model=model_name)

with st.sidebar:
    st.header("Paramètres")
    model_choice = st.selectbox(
        "Modèle",
        options=list(MODEL_DISPLAY_NAMES.keys()),
        format_func=lambda x: MODEL_DISPLAY_NAMES[x]
    )
    clf = load_pipeline(model_choice)
    st.success("Modèle prêt")

LABEL_MAPS = {
    "cardiffnlp/twitter-xlm-roberta-base-sentiment": {
        "negative": "Négatif", "neutral": "Neutre", "positive": "Positif",
        "NEGATIVE": "Négatif", "NEUTRAL": "Neutre", "POSITIVE": "Positif",
    },
    "distilbert-base-uncased-finetuned-sst-2-english": {
        "NEGATIVE": "Négatif", "POSITIVE": "Positif",
    }
}

def normalize_label(label: str) -> str:
    return LABEL_MAPS.get(model_choice, {}).get(label, label)

@st.cache_data(show_spinner=False)
def run_inference_text(text: str):
    res = clf(text)[0]
    return {"label": normalize_label(res["label"]), "score": float(res["score"])}

@st.cache_data(show_spinner=True)
def run_inference_batch(texts: list[str]):
    out = []
    BATCH = 32
    total = len(texts)
    progress_bar = st.progress(0)         
    progress_text = st.empty()            

    for i in range(0, total, BATCH):
        chunk = texts[i:i+BATCH]
        preds = clf(chunk)
        for t, p in zip(chunk, preds):
            out.append({
                "text": t,
                "label": normalize_label(p["label"]),
                "score": float(p["score"])
            })

        percent_done = min(100, round(((i + BATCH) / total) * 100, 1))
        progress_bar.progress(percent_done / 100)
        progress_text.text(f"Traitement : {min(i + BATCH, total)}/{total} ({percent_done} %)")

    progress_text.text("✅ Analyse terminée")
    return pd.DataFrame(out)


tab_single, tab_batch = st.tabs(["Analyse unitaire", "Analyse par fichier"])

# === TAB 1 : Single ===
with tab_single:
    st.subheader("Analyse d’un texte")
    sample = st.selectbox(
        "Exemples",
        [
            "J'adore ce produit, la qualité est excellente et le service parfait !",
            "Le film est moyen, pas terrible mais pas catastrophique non plus.",
            "C'est une perte de temps, très déçu.",
            "Le service client a été très réactif, je suis satisfait.",
            "La livraison est arrivée en retard et le produit était abîmé."
        ],
        index=0
    )
    text_input = st.text_area("Texte à analyser", value=sample, height=120)

    if st.button("Analyser"):
        if not text_input.strip():
            st.warning("Veuillez saisir un texte.")
        else:
            res = run_inference_text(text_input.strip())
            st.metric("Sentiment", res["label"])
            score_pct = round(res["score"] * 100, 1)
            st.progress(res["score"])
            st.write(f"**Confiance :** {score_pct}%")

# === TAB 2 : Batch ===
with tab_batch:
    st.subheader("Analyse d’un fichier CSV")
    file = st.file_uploader("Importer un CSV", type=["csv"])
    if file is not None:
        df = pd.read_csv(file)
        st.write("Aperçu du CSV :", df.head())
        text_col = st.selectbox("Colonne texte", options=df.columns.tolist())

        if st.button("Lancer l’analyse par lot"):
            texts = df[text_col].astype(str).fillna("").tolist()
            if not any(t.strip() for t in texts):
                st.error("Colonne vide ou invalide.")
            else:
                res_df = run_inference_batch(texts)
                st.write("Aperçu des résultats :", res_df.head())

                # Graphique 1 : Répartition des sentiments
                counts = res_df["label"].value_counts().reset_index()
                counts.columns = ["Sentiment", "Nombre"]
                fig_pie = px.pie(counts, names="Sentiment", values="Nombre", title="Répartition des sentiments")
                st.plotly_chart(fig_pie, use_container_width=True)

                # Graphique 2 : Distribution des scores
                fig_hist = px.histogram(res_df, x="score", color="label", nbins=20, title="Distribution des scores")
                st.plotly_chart(fig_hist, use_container_width=True)

                # Téléchargement du résultat
                buffer = io.StringIO()
                res_df.to_csv(buffer, index=False)
                st.download_button(
                    label="Télécharger résultats (CSV)",
                    data=buffer.getvalue(),
                    file_name="sentiments_results.csv",
                    mime="text/csv",
                )
