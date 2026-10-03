# 🛒 Iris E-Commerce Price Prediction — Morocco

> Plateforme complète de scraping, analyse et prédiction de prix pour le e-commerce marocain (Iris.ma)

##  Overview
Projet end-to-end de Price Intelligence : collecte automatisée → nettoyage → ML → Data Warehouse → Dashboard BI

**Stack :** Python · Selenium · PostgreSQL · PyTorch · HuggingFace · XGBoost · LightGBM · Snowflake · Power BI

##  Résultats ML
| Métrique | Valeur |
|---------|--------|
| R² Global | **0.7957** |
| MAE Global | **1,338 DH** |
| R² ≤10K DH | 0.6568 |
| R² >10K DH | 0.2130 |

## 🔗 Links
- 📊 **Dashboard Power BI :** https://app.powerbi.com/groups/me/reports/65c6cffa-a80b-40fc-9d07-bd8ed2894478/50b91217e421bd3524f8
- 🤖 **Demo API :** *(coming soon)*

##  Pipeline
1. Web Scraping (Selenium) → MySQL
2. EDA & Cleaning (Jupyter) → PostgreSQL
3. ML Training : BERT + XGBoost + LightGBM (Kaggle GPU)
4. Data Warehouse : Snowflake Star Schema
5. Dashboard : Power BI (7 pages)
6. API : FastAPI *(coming soon)*
