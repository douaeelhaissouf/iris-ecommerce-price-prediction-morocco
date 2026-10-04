from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional
from predict import predict_price, get_models

# ── App ──────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Iris Price Prediction API",
    description=(
        "API de prédiction de prix pour les produits e-commerce marocains. "
        "Modèle : BERT + XGBoost + LightGBM (ensemble deux segments).\n\n"
        "**Source des données** : Iris.ma — 15 071 produits scrapés.\n"
        "**Performances** : R²=0.7957 | MAE=1 338 DH"
    ),
    version="1.0.0",
    docs_url="/",          # Swagger UI à la racine
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Schémas ──────────────────────────────────────────────────────────────────
class PredictRequest(BaseModel):
    title: str = Field(..., example="Samsung Galaxy A54 5G 128Go Noir")
    category: str = Field(..., example="Téléphones & Tablettes")
    subcategory: str = Field(..., example="Smartphones")
    brand: str = Field(..., example="Samsung")
    segment: Optional[str] = Field(
        None,
        description="Forcer le segment : 'low' (≤10 000 DH) ou 'high' (>10 000 DH). "
                    "Laisser vide pour détection automatique.",
        example=None,
    )

class PredictResponse(BaseModel):
    predicted_price_dh: float
    segment: str
    xgb_price: float
    lgbm_price: float
    currency: str = "MAD"

class HealthResponse(BaseModel):
    status: str
    models_loaded: bool

# ── Routes ───────────────────────────────────────────────────────────────────
@app.get("/health", response_model=HealthResponse, tags=["Monitoring"])
def health():
    """Vérifie que l'API tourne et si les modèles sont en cache."""
    from predict import _models
    return {"status": "ok", "models_loaded": bool(_models)}


@app.post("/predict", response_model=PredictResponse, tags=["Prédiction"])
def predict(req: PredictRequest):
    """
    Prédit le prix (en DH) d'un produit e-commerce marocain.

    - **title** : Titre complet du produit
    - **category** : Catégorie principale (ex. *Informatique*)
    - **subcategory** : Sous-catégorie (ex. *Laptops*)
    - **brand** : Marque (ex. *Lenovo*)
    - **segment** : Optionnel — `low` (≤10 000 DH) ou `high` (>10 000 DH)
    """
    try:
        result = predict_price(
            title=req.title,
            category=req.category,
            subcategory=req.subcategory,
            brand=req.brand,
            segment_hint=req.segment,
        )
        return {**result, "currency": "MAD"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Chargement des modèles au démarrage (évite le cold-start sur la 1ère requête)
@app.on_event("startup")
async def startup_event():
    get_models()
