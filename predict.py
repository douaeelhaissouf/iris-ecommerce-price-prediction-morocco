import os
import json
import torch
import pickle
import numpy as np
from pathlib import Path
from huggingface_hub import hf_hub_download, snapshot_download
from sentence_transformers import SentenceTransformer

# ── Config ──────────────────────────────────────────────────────────────────
HF_REPO = "douaeelhaissouf/iris-price-models"
MODEL_DIR = Path("models_cache")
MODEL_DIR.mkdir(exist_ok=True)

THRESHOLD = 10_000   # DH — sépare segment low / high

# ── Chargement paresseux (une seule fois au démarrage) ──────────────────────
_models = {}

def _load_file(filename: str) -> Path:
    """Télécharge depuis HF si pas déjà en cache local."""
    local = MODEL_DIR / filename
    if not local.exists():
        hf_hub_download(repo_id=HF_REPO, filename=filename, local_dir=MODEL_DIR)
    return local

def _load_sbert() -> SentenceTransformer:
    """Charge le modèle SBERT (encodeur de texte)."""
    sbert_dir = MODEL_DIR / "sbert"
    if not sbert_dir.exists():
        snapshot_download(repo_id=HF_REPO, local_dir=MODEL_DIR,
                          allow_patterns=["sbert/*"])
    return SentenceTransformer(str(sbert_dir))

def _load_bert_weights(segment: str) -> dict:
    """Charge les poids fine-tunés BERT (low ou high)."""
    filename = f"bert_{segment}.pt"
    path = _load_file(filename)
    return torch.load(path, map_location="cpu", weights_only=True)

def _load_pkl(filename: str):
    with open(_load_file(filename), "rb") as f:
        return pickle.load(f)

def get_models():
    """Charge tous les modèles une seule fois et les met en cache mémoire."""
    global _models
    if _models:
        return _models

    print("⏳ Chargement des modèles depuis Hugging Face…")

    sbert = _load_sbert()

    # Poids BERT fine-tunés  → dictionnaire de scalers / couches
    bert_low_w  = _load_bert_weights("low")
    bert_high_w = _load_bert_weights("high")

    xgb_low   = _load_pkl("xgb_low.pkl")
    xgb_high  = _load_pkl("xgb_high.pkl")
    lgbm_low  = _load_pkl("lgbm_low.pkl")
    lgbm_high = _load_pkl("lgbm_high.pkl")
    te_low    = _load_pkl("te_low.pkl")
    te_high   = _load_pkl("te_high.pkl")

    with open(_load_file("poids.json")) as f:
        poids = json.load(f)   # {"low": [w_xgb, w_lgbm], "high": [w_xgb, w_lgbm]}

    _models = dict(
        sbert=sbert,
        bert_low_w=bert_low_w,
        bert_high_w=bert_high_w,
        xgb_low=xgb_low, xgb_high=xgb_high,
        lgbm_low=lgbm_low, lgbm_high=lgbm_high,
        te_low=te_low, te_high=te_high,
        poids=poids,
    )
    print("✅ Modèles chargés !")
    return _models


# ── Pipeline de prédiction ───────────────────────────────────────────────────

def encode_text(text: str, sbert: SentenceTransformer) -> np.ndarray:
    """Encode un texte en vecteur SBERT (384 dims)."""
    return sbert.encode([text], normalize_embeddings=True)[0]


def build_features(
    title: str,
    category: str,
    subcategory: str,
    brand: str,
    segment: str,
    models: dict,
) -> np.ndarray:
    """
    Construit le vecteur de features :
      - embedding SBERT du titre (384 dims)
      - target encoding de category + subcategory + brand
    """
    sbert = models["sbert"]
    te    = models[f"te_{segment}"]

    text_vec = encode_text(title, sbert)   # (384,)

    # Target encoding des variables catégorielles
    cat_df_like = [[category, subcategory, brand]]
    import pandas as pd
    df = pd.DataFrame(cat_df_like, columns=["category", "subcategory", "brand"])
    cat_encoded = te.transform(df).values[0]   # (3,)

    return np.concatenate([text_vec, cat_encoded])   # (387,)


def predict_price(
    title: str,
    category: str,
    subcategory: str,
    brand: str,
    segment_hint: str | None = None,
) -> dict:
    """
    Prédit le prix d'un produit marocain.

    segment_hint : "low" | "high" | None (auto-détecté via une première passe low)
    """
    models = get_models()
    poids  = models["poids"]

    # ── Détermination du segment ─────────────────────────────────────────────
    if segment_hint is None:
        # Passe rapide sur le segment low pour estimer
        feats_low = build_features(title, category, subcategory, brand, "low", models)
        X = feats_low.reshape(1, -1)
        p_xgb  = models["xgb_low"].predict(X)[0]
        p_lgbm = models["lgbm_low"].predict(X)[0]
        w = poids["low"]
        rough_price = w[0] * p_xgb + w[1] * p_lgbm
        segment = "high" if rough_price > THRESHOLD else "low"
    else:
        segment = segment_hint

    # ── Features du bon segment ──────────────────────────────────────────────
    feats = build_features(title, category, subcategory, brand, segment, models)
    X     = feats.reshape(1, -1)

    xgb_pred  = models[f"xgb_{segment}"].predict(X)[0]
    lgbm_pred = models[f"lgbm_{segment}"].predict(X)[0]

    w     = poids[segment]
    price = float(w[0] * xgb_pred + w[1] * lgbm_pred)

    return {
        "predicted_price_dh": round(price, 2),
        "segment": segment,
        "xgb_price":  round(float(xgb_pred), 2),
        "lgbm_price": round(float(lgbm_pred), 2),
    }
