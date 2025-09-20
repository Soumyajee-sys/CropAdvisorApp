# Crop Advisor App

A machine learning-powered mobile application that provides personalized crop recommendations to farmers based on soil, location, weather, and dynamic mandi price data. Built with a modular Python FastAPI backend and a cross-platform React Native frontend.

---

## Features

- District-level crop recommendation using geolocation (GPS)
- Integrates local soil data, seasonal crop calendars, and real-time mandi (market) price datasets
- Machine learning model predicts the most suitable and profitable crops
- Weather and market data APIs enable real-time, adaptive suggestions
- User-friendly mobile app (React Native with Expo), tested on Android
- Displays expected revenue, water requirements, duration, and key info per recommended crop
- Uses rank correlation and AI to optimize feature selection and accuracy

---

## Tech Stack

- **Backend:** Python, FastAPI, Pandas, Scikit-learn, Joblib
- **Frontend:** React Native, Expo SDK
- **APIs & Libraries:** Expo Location, OpenStreetMap Nominatim, Axios
- **Data:** CSV-based soil/crop datasets, real mandi price records
- **Dev Tools:** Git, Visual Studio Code, Android Studio, same-network testing

---

## Setup and Usage

### Prerequisites

- Python 3.8+
- Node.js and npm
- Expo CLI (`npm install -g expo-cli`)
- Android device (for best results) on same Wi-Fi as backend server

### Backend

1. Clone the repository and install Python dependencies (`pip install -r requirements.txt`)
2. Place `crop_prediction_model.pkl`, `crop_label_encoders.pkl`, and dataset `.csv` files in the root directory
3. Run the FastAPI server:

---

## How It Works

- User opens mobile app and grants location permissions
- App reverse geocodes GPS location to find the administrative district
- Sends request to FastAPI backend for that district, along with real-time lat/lon/season info
- Backend loads relevant soil, calendar, and price data; runs machine learning model (informed by feature rank correlation)
- Backend responds with top 5 crop recommendations and supporting details
- App displays actionable guidance to user in a simple card format

---

## Research and References

- [FastAPI documentation](https://fastapi.tiangolo.com/)
- [React Native docs](https://reactnative.dev/)
- [OpenStreetMap Nominatim](https://nominatim.org/release-docs/latest/api/Reverse/)
- [Pandas documentation](https://pandas.pydata.org/docs/)

