
from fastapi import FastAPI
from pydantic import BaseModel
import pickle
from pathlib import Path
import pandas as pd

app = FastAPI(title="Job Market Analysis API", version="0.1.0")

MODEL_PATH = Path("ml/models/salary_model.pkl")
_model = None

def get_model():
    global _model
    if _model is None:
        try:
            if MODEL_PATH.exists():
                import joblib
                loaded_data = joblib.load(MODEL_PATH)
                # Ensure we have a dictionary, not a numpy array
                if isinstance(loaded_data, dict):
                    _model = loaded_data
                else:
                    _model = {"status": "error", "error": "Model format incorrect"}
            else:
                _model = {"status": "missing"}
        except Exception as e:
            _model = {"status": "error", "error": str(e)}
    return _model

class SalaryRequest(BaseModel):
    years_experience: float
    skills: list[str] = []
    currency: str = "USD"
    period: str = "YEARLY"
    location: str | None = None
    job_title: str | None = None

class SkillForecastRequest(BaseModel):
    skill: str
    horizon_months: int = 12

class CareerPlanRequest(BaseModel):
    current_skills: list[str]
    target_role: str | None = None

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/predict-salary")
def predict_salary(req: SalaryRequest):
    model = get_model()
    if model.get("status") == "missing":
        return {
            "predicted_salary": None,
            "lower_bound": None,
            "upper_bound": None,
            "message": "Model not found. Run src/ml/salary_prediction_model.py",
        }
    elif model.get("status") == "error":
        return {
            "predicted_salary": None,
            "lower_bound": None,
            "upper_bound": None,
            "message": f"Model error: {model.get('error', 'Unknown error')}",
        }
    elif model.get("status") != "trained":
        return {
            "predicted_salary": None,
            "lower_bound": None,
            "upper_bound": None,
            "message": "Model not trained yet. Run src/ml/salary_prediction_model.py",
        }
    
    try:
        # Prepare features for the model
        features_dict = {}
        
        # Currency encoding
        if 'currency' in model.get('label_encoders', {}):
            currency_encoder = model['label_encoders']['currency']
            try:
                features_dict['currency_encoded'] = currency_encoder.transform([req.currency])[0]
            except ValueError:
                # If currency not in encoder, use 0 (USD)
                features_dict['currency_encoded'] = 0
        
        # Period encoding
        if 'period' in model.get('label_encoders', {}):
            period_encoder = model['label_encoders']['period']
            try:
                features_dict['period_encoded'] = period_encoder.transform([req.period])[0]
            except ValueError:
                # If period not in encoder, use 0 (YEARLY)
                features_dict['period_encoded'] = 0
        
        # Add job_id_length feature (simulate based on years experience)
        features_dict['job_id_length'] = min(max(int(req.years_experience * 2), 5), 20)
        
        # Prepare feature vector
        features = []
        for col in model.get('feature_columns', []):
            features.append(features_dict.get(col, 0))
        
        # Scale features
        scaler = model.get('scaler')
        if scaler:
            features_scaled = scaler.transform([features])
        else:
            features_scaled = [features]
        
        # Make prediction
        mean_model = model["mean_model"]
        pred = float(mean_model.predict(features_scaled)[0])
        
        # Add some variance for bounds (simplified)
        variance = pred * 0.2  # 20% variance
        lo = max(pred - variance, pred * 0.7)  # At least 70% of prediction
        hi = pred + variance
        
        return {
            "predicted_salary": pred,
            "lower_bound": lo,
            "upper_bound": hi,
            "message": "OK",
            "model_metrics": model.get("metrics", {}),
        }
    except Exception as e:
        return {
            "predicted_salary": None,
            "lower_bound": None,
            "upper_bound": None,
            "message": f"Prediction error: {str(e)}",
        }

@app.post("/forecast-skills")
def forecast_skills(req: SkillForecastRequest):
    # Load precomputed forecasts and filter
    path = Path("data/gold/skill_forecasts")
    if not path.exists():
        return {"message": "Forecasts not found. Run src/ml/skill_forecasting.py"}
    try:
        df = pd.read_parquet(path)
    except Exception:
        return {"message": "Unable to read forecasts parquet"}
    out = df[df.get("skill","").str.lower() == req.skill.lower()].copy()
    if out.empty:
        # Return top few skills available
        top = (
            df["skill"].value_counts().head(10).index.tolist() if "skill" in df.columns else []
        )
        return {"skill": req.skill, "forecast": [], "available_skills": top}
    # Limit horizon
    out = out.sort_values("date").head(req.horizon_months)
    return {
        "skill": req.skill,
        "forecast": [
            {"date": str(d), "forecast": float(v)} for d, v in zip(out["date"], out["forecast"])
        ],
    }

@app.post("/career-recommendations")
def career_recommendations(req: CareerPlanRequest):
    # Heuristic: suggest skills from top-demand skills that are not in current set
    skills_path = Path("data/gold/skills_demand.parquet")
    
    if skills_path.exists():
        try:
            df = pd.read_parquet(skills_path)
        except Exception:
            df = None
    else:
        df = None
    
    if df is None or df.empty or not {"skill","count"}.issubset(df.columns):
        return {"message": "Skills demand not available. Run Spark ETL."}
    have = set(s.lower() for s in req.current_skills)
    candidates = (
        df.sort_values("count", ascending=False)["skill"].astype(str).str.lower().tolist()
    )
    recs = [s for s in candidates if s not in have][:20]
    return {
        "target_role": req.target_role,
        "recommended_skills": recs,
        "note": "Prototype heuristic; next step: role similarity graph and salary premium weighting.",
    }
