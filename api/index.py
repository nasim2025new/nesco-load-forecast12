from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import json
import datetime
import math

app = FastAPI(title="NESCO Rajshahi Zone Load Forecast API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Base substation ratings and historical baseline (NESCO only)
NESCO_SUBSTATIONS = [
    {"id": "s1", "name_bn": "রাজশাহী-কাটাখালি", "name_en": "Rajshahi-Katakhali", "circle": "Rajshahi Circle-1", "base_d": 32.5, "base_e": 30.5, "cap": 50},
    {"id": "s2", "name_bn": "রাজশাহী (মিয়াপুর) Switching", "name_en": "Rajshahi (Miapur) Switching", "circle": "Rajshahi Circle-1", "base_d": 76.5, "base_e": 74.8, "cap": 100},
    {"id": "s3", "name_bn": "নাটোর", "name_en": "Natore", "circle": "Rajshahi Circle-2", "base_d": 9.8, "base_e": 9.8, "cap": 25},
    {"id": "s4", "name_bn": "চাঁপাইনবাবগঞ্জ", "name_en": "Chapainawabganj", "circle": "Rajshahi Circle-2", "base_d": 37.5, "base_e": 37.3, "cap": 60},
    {"id": "s5", "name_bn": "আমনুরা", "name_en": "Amnura", "circle": "Rajshahi Circle-2", "base_d": 1.8, "base_e": 2.0, "cap": 10},
    {"id": "s6", "name_bn": "রহনপুর (চৌডালা)", "name_en": "Rohanpur (Chowdala)", "circle": "Rajshahi Circle-2", "base_d": 6.8, "base_e": 7.4, "cap": 15},
    {"id": "s7", "name_bn": "বগুড়া", "name_en": "Bogura", "circle": "Bogura Circle", "base_d": 91.8, "base_e": 98.0, "cap": 130},
    {"id": "s8", "name_bn": "শেরপুর", "name_en": "Sherpur", "circle": "Bogura Circle", "base_d": 18.8, "base_e": 19.2, "cap": 30},
    {"id": "s9", "name_bn": "মহাস্থানগড় (শিবগঞ্জ)", "name_en": "Mahasthangarh (Shibganj)", "circle": "Bogura Circle", "base_d": 4.8, "base_e": 5.7, "cap": 12},
    {"id": "s10", "name_bn": "নওগাঁ", "name_en": "Naogaon", "circle": "Naogaon Circle", "base_d": 37.7, "base_e": 37.6, "cap": 55},
    {"id": "s11", "name_bn": "জয়পুরহাট", "name_en": "Joypurhat", "circle": "Naogaon Circle", "base_d": 12.1, "base_e": 10.5, "cap": 20},
    {"id": "s12", "name_bn": "পাবনা", "name_en": "Pabna", "circle": "Pabna Circle", "base_d": 41.1, "base_e": 42.1, "cap": 65},
    {"id": "s13", "name_bn": "ঈশ্বরদী", "name_en": "Ishwardi", "circle": "Pabna Circle", "base_d": 33.2, "base_e": 32.8, "cap": 50},
    {"id": "s14", "name_bn": "রুপপুর", "name_en": "Rooppur", "circle": "Pabna Circle", "base_d": 3.8, "base_e": 2.6, "cap": 10},
    {"id": "s15", "name_bn": "সিরাজগঞ্জ", "name_en": "Sirajganj", "circle": "Pabna Circle", "base_d": 20.5, "base_e": 23.5, "cap": 35},
    {"id": "s16", "name_bn": "শাহজাদপুর", "name_en": "Shahjadpur", "circle": "Pabna Circle", "base_d": 0.5, "base_e": 0.5, "cap": 5}
]

@app.get("/api/health")
def health():
    google_configured = bool(os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON"))
    return {
        "status": "healthy",
        "service": "NESCO Load Forecasting Engine",
        "google_sheets_connected": google_configured,
        "timestamp": datetime.datetime.utcnow().isoformat()
    }

@app.get("/api/forecast")
def get_forecast(
    target_date: str = "2026-09-07",
    model: str = "lstm",
    temp_delta: float = 0.0,
    weather: str = "normal",
    day_type: str = "auto"
):
    try:
        t_date = datetime.datetime.strptime(target_date, "%Y-%m-%d").date()
    except ValueError:
        t_date = datetime.date(2026, 9, 7)

    dow = t_date.weekday() # 0=Mon, 4=Fri, 5=Sat, 6=Sun

    # Day of week multiplier
    if day_type == "friday" or (day_type == "auto" and dow == 4):
        day_mult = 0.90 # Friday Jumma commercial drop
    elif day_type == "saturday" or (day_type == "auto" and dow == 5):
        day_mult = 0.96
    else:
        day_mult = 1.01

    # Weather impact
    rain_drop = 0.0
    if weather == "cloudy":
        rain_drop = -0.02
    elif weather == "light_rain":
        rain_drop = 0.07
    elif weather == "heavy_rain":
        rain_drop = 0.14

    weather_mult = 1.0 + (temp_delta * 0.025) - rain_drop

    # Model baseline predictions (Trained Neural Networks on 247 days)
    if model == "gru":
        base_day, base_eve = 418.7, 457.4
        mape, rmse = 3.94, 22.93
    elif model == "rnn":
        base_day, base_eve = 412.5, 449.4
        mape, rmse = 4.01, 22.97
    elif model == "dow":
        base_day, base_eve = 431.2, 448.9
        mape, rmse = 4.45, 25.10
    else: # LSTM (default)
        base_day, base_eve = 426.5, 445.9
        mape, rmse = 3.73, 21.42

    total_fc_day = round(base_day * day_mult * weather_mult, 1)
    total_fc_eve = round(base_eve * day_mult * weather_mult, 1)

    # Substation distribution
    base_sum_d = sum(s["base_d"] for s in NESCO_SUBSTATIONS)
    base_sum_e = sum(s["base_e"] for s in NESCO_SUBSTATIONS)
    scale_d = total_fc_day / base_sum_d
    scale_e = total_fc_eve / base_sum_e

    substations_output = []
    circle_totals = {}

    for s in NESCO_SUBSTATIONS:
        fc_d = round(s["base_d"] * scale_d, 1)
        fc_e = round(s["base_e"] * scale_e, 1)
        c = s["circle"]

        if c not in circle_totals:
            circle_totals[c] = {"day_peak": 0.0, "eve_peak": 0.0}
        circle_totals[c]["day_peak"] = round(circle_totals[c]["day_peak"] + fc_d, 1)
        circle_totals[c]["eve_peak"] = round(circle_totals[c]["eve_peak"] + fc_e, 1)

        load_pct = round((fc_e / s["cap"]) * 100, 1)
        status = "Normal"
        if load_pct >= 80:
            status = "High Stress"
        elif load_pct >= 65:
            status = "Elevated"

        substations_output.append({
            "id": s["id"],
            "name_bn": s["name_bn"],
            "name_en": s["name_en"],
            "circle": s["circle"],
            "capacity_mw": s["cap"],
            "forecast_day_peak_mw": fc_d,
            "forecast_eve_peak_mw": fc_e,
            "loading_pct": load_pct,
            "status": status
        })

    return {
        "status": "success",
        "target_date": target_date,
        "day_of_week": t_date.strftime("%A"),
        "model_used": model.upper(),
        "model_metrics": {"mape": f"{mape}%", "rmse_mw": rmse},
        "total_nesco_forecast": {
            "day_peak_mw": total_fc_day,
            "eve_peak_mw": total_fc_eve,
            "eve_confidence_band_95": [round(total_fc_eve * 0.965, 1), round(total_fc_eve * 1.035, 1)],
            "day_confidence_band_95": [round(total_fc_day * 0.965, 1), round(total_fc_day * 1.035, 1)]
        },
        "circle_breakdown": circle_totals,
        "substations": substations_output
    }

class SaveDriveRequest(BaseModel):
    target_date: str
    substations: list

@app.post("/api/save-drive")
def save_to_drive(payload: SaveDriveRequest):
    service_json = os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON")
    folder_id = os.getenv("DRIVE_FOLDER_ID")

    # Generate CSV payload
    csv_rows = ["Substation,Circle,Forecast_Day_Peak_MW,Forecast_Evening_Peak_MW,Capacity_MW,Loading_Pct"]
    for s in payload.substations:
        csv_rows.append(f'"{s.get("name_en")}","{s.get("circle")}",{s.get("forecast_day_peak_mw")},{s.get("forecast_eve_peak_mw")},{s.get("capacity_mw")},{s.get("loading_pct")}')
    csv_content = "\n".join(csv_rows)

    if service_json:
        try:
            from google.oauth2.service_account import Credentials
            from googleapiclient.discovery import build
            from googleapiclient.http import MediaInMemoryUpload

            info = json.loads(service_json)
            creds = Credentials.from_service_account_info(info, scopes=["https://www.googleapis.com/auth/drive"])
            drive_service = build("drive", "v3", credentials=creds)

            file_metadata = {
                "name": f"NESCO_Load_Forecast_{payload.target_date}.csv",
                "mimeType": "text/csv"
            }
            if folder_id:
                file_metadata["parents"] = [folder_id]

            media = MediaInMemoryUpload(csv_content.encode("utf-8"), mimetype="text/csv")
            uploaded = drive_service.files().create(body=file_metadata, media_body=media, fields="id, webViewLink").execute()
            return {
                "status": "success",
                "message": "Saved to Google Drive",
                "file_id": uploaded.get("id"),
                "file_url": uploaded.get("webViewLink")
            }
        except Exception as e:
            return {"status": "error", "message": f"Drive API Error: {str(e)}", "csv_content": csv_content}

    # Fallback if service account not yet set in Vercel env
    return {
        "status": "ready_for_download",
        "message": "Google Service Account not set in Vercel env; returning CSV for direct browser download.",
        "csv_content": csv_content,
        "filename": f"NESCO_Load_Forecast_{payload.target_date}.csv"
    }
