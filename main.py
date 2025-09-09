from fastapi import FastAPI, Query
import pandas as pd
import joblib
from weather import get_seasonal_forecast
import numpy as np

app = FastAPI()

# Load datasets with lowercase column names for general data processing
soil_df = pd.read_csv("crop_prediction_dataset.csv")
soil_df.columns = soil_df.columns.str.lower()
soil_df['district_lower'] = soil_df['district'].str.strip().str.lower()

calendar_df = pd.read_csv("crop_calendar.csv")
calendar_df.columns = calendar_df.columns.str.lower()
calendar_df['district_lower'] = calendar_df['district'].str.strip().str.lower()

mandi_prices_df = pd.read_csv("mandi_price.csv")
mandi_prices_df.columns = mandi_prices_df.columns.str.lower()
mandi_prices_df = mandi_prices_df[['district', 'commodity', 'price (rs/quintal)', 'season']]
mandi_prices_df['district_lower'] = mandi_prices_df['district'].str.strip().str.lower()

# Load model and encoders - these expect original casing of features
model = joblib.load("crop_prediction_model.pkl")
encoders = joblib.load("crop_label_encoders.pkl")

# Config mappings with lowercase keys consistent with dataset
CROP_MAPPING = {
    'rice': 'Rice (Paddy)',
    'maize': 'Maize',
    'wheat': 'Wheat',
    'pulses': 'Arhar (Tur)',
    'sugarcane': 'Sugarcane'
}
YIELD_ESTIMATES = {
    'rice': 25,
    'maize': 18,
    'wheat': 20,
    'pulses': 8,
    'sugarcane': 300
}
MONTH_MAPPING = {m: i + 1 for i, m in enumerate(['january','february','march','april','may','june','july','august','september','october','november','december'])}

def month_to_int(m):
    return MONTH_MAPPING.get(m.lower(), 1)

def get_ai_confidence(soil_lower):
    soil = {
        'Soil_Type': soil_lower.get('soil_type'),
        'Soil_pH': soil_lower.get('soil_ph'),
        'N': soil_lower.get('n'),
        'P': soil_lower.get('p'),
        'K': soil_lower.get('k'),
        'Soil_Texture': soil_lower.get('soil_texture'),
        'Irrigation_Type': soil_lower.get('irrigation_type'),
        'District': soil_lower.get('district')
    }
    feats = pd.DataFrame([soil])
    for col, le in encoders.items():
        if col in feats.columns:
            feats[col] = le.transform(feats[col])
    probs = model.predict_proba(feats)[0]
    return {c.lower(): float(probs[i]) for i, c in enumerate(model.classes_)}

def get_price(district, commodity, season):
    df = mandi_prices_df
    row = df[(df.district_lower == district.lower()) & (df.commodity == commodity) & (df.season == season)]
    if not row.empty:
        return float(row['price (rs/quintal)'].iloc[0])
    row = df[(df.district_lower == district.lower()) & (df.commodity == commodity)]
    if not row.empty:
        return float(row['price (rs/quintal)'].mean())
    return float(df[df.commodity == commodity]['price (rs/quintal)'].mean())

def profitability(price, commodity):
    ranges = mandi_prices_df.groupby('commodity')['price (rs/quintal)'].agg(['min', 'max']).to_dict()
    min_prices = ranges['min']
    max_prices = ranges['max']
    mn = min_prices.get(commodity, None)
    mx = max_prices.get(commodity, None)
    if mn is None or mx is None or mx == mn:
        return 0.0
    return round((price - mn) / (mx - mn) * 100, 1)

def build_recommendations(district, lat, lon, season_filter, ai_weight):
    soil_rows = soil_df[soil_df.district_lower == district.lower()]
    if soil_rows.empty:
        return None, None  # no soil data found
    soil_row = soil_rows.iloc[0]

    soil_lower = {col: soil_row[col] for col in soil_df.columns if col != 'district_lower'}
    soil_lower['district'] = soil_row['district']

    ai_conf = get_ai_confidence(soil_lower)
    cal = calendar_df[calendar_df.district_lower == district.lower()]
    if season_filter.lower() != 'all':
        cal = cal[cal.season.str.lower() == season_filter.lower()]

    if cal.empty:
        return None, None  # no calendar data found

    recs = []
    for _, r in cal.iterrows():
        crop = r.crop.lower()
        if crop not in CROP_MAPPING:
            continue
        comm = CROP_MAPPING[crop]
        price = get_price(district.lower(), comm, r.season)
        prof = profitability(price, comm)
        ai_c = ai_conf.get(crop, 0)
        combined = round((ai_weight * ai_c * 100 + (1 - ai_weight) * prof), 1)

        recs.append({
            'crop': r.crop,
            'season': r.season,
            'ai_confidence': round(ai_c * 100, 1),
            'combined_score': combined,
            'recommendation_level': 'Highly Recommended' if combined >= 75 else 'Recommended' if combined >= 60 else 'May Consider',
            'mandi_price': round(price, 2),
            'expected_revenue': round(price * YIELD_ESTIMATES[crop], 2),
            'water_requirement': r.water_requirement,
            'growth_duration_days': r.growth_duration_days
        })
    recs = sorted(recs, key=lambda x: x['combined_score'], reverse=True)
    summary = {
        'total_crops_analyzed': len(recs),
        'top_recommended_crop': recs[0]['crop'] if recs else None,
        'top_recommendation_score': recs[0]['combined_score'] if recs else None
    }
    return summary, recs[:5]

@app.get("/recommend")
def recommend(district: str = Query(...), lat: float = Query(...), lon: float = Query(...),
              season: str = Query('all'), ai_weight: float = Query(0.6)):
    summary, recs = build_recommendations(district, lat, lon, season, ai_weight)
    if summary is None or recs is None:
        return {"error": f"No data found for district '{district}'. Please check the district name or data availability."}
    return {
        'district': district,
        'season_filter': season,
        'ai_weight_used': ai_weight,
        'market_weight_used': round(1 - ai_weight, 2),
        'summary': summary,
        'recommendations': recs
    }
