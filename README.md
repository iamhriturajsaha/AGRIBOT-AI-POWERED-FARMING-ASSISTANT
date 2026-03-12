# 🌱AgriBot — AI-Powered Farming Assistant

An intelligent agricultural decision support system that helps farmers make data-driven decisions using machine learning and computer vision.

## Overview

Agriculture is central to global food security, yet farmers routinely face decisions without adequate data — which crop to plant, how much yield to expect or whether a plant is diseased. AgriBot addresses these challenges by integrating three AI-powered modules into a single, easy-to-use interface.

**What AgriBot can do -**

- Recommend the most suitable crop based on soil and climate conditions.
- Predict expected crop yield from historical production data.
- Detect tomato plant diseases from leaf images using deep learning.

## Features

| Module | Inputs | Output |
|---|---|---|
| 🌱 **Crop Recommendation** | N, P, K, Temperature, Humidity, pH, Rainfall | Recommended crop |
| 📊 **Yield Prediction** | Crop type, State, Season, Area | Estimated production |
| 🍅 **Disease Detection** | Leaf image (tomato) | Disease class + confidence |

## System Architecture

```
┌─────────────────────────────────────┐
│         Streamlit Web Interface     │
└─────┬───────────┬──────────────┬────┘
      │           │              │
      ▼           ▼              ▼
┌──────────┐ ┌─────────┐ ┌──────────────┐
│  Crop    │ │  Yield  │ │   Disease    │
│  Model   │ │  Model  │ │   CNN Model  │
└──────────┘ └─────────┘ └──────────────┘
      │           │              │
      └───────────┴──────────────┘
                  │
                  ▼
       Trained Models + Datasets
```

## Modules

### 1. Crop Recommendation

Predicts the most suitable crop to grow based on soil nutrients and environmental conditions.

- **Inputs -** Nitrogen (N), Phosphorus (P), Potassium (K), Temperature, Humidity, pH, Rainfall.

- **Pipeline -** Data preprocessing → Feature scaling → Label encoding → Model prediction.

- **Saved artifacts -** `crop_recommendation_model.pkl`, `crop_scaler.pkl`, `crop_label_encoder.pkl`

### 2. Crop Yield Prediction

Forecasts expected crop production using historical agricultural datasets.

- **Inputs -** Crop type, State, Season, Cultivation area.

- **Pipeline -** Data cleaning → Feature engineering → Categorical encoding → Model prediction.

- **Saved artifacts -** `yield_prediction_model.pkl`, `yield_feature_columns.pkl`

### 3. Plant Disease Detection

Uses a Convolutional Neural Network (CNN) to classify tomato leaf diseases from uploaded images.

- **Supported disease classes -**
  - Tomato Leaf Mold.
  - Tomato Mosaic Virus.
  - Early Blight.
  - Late Blight.
  - Healthy Leaf.
  - Bacterial Spot.
  - Septoria Keaf Spot.
  - Spider Mites.
  - Target Spot.
  - Yellow Leaf Curl Virus.

- **Pipeline -** Image upload → Preprocessing → Model inference → Disease classification.

- **Saved artifacts -** `best_tomato_model.h5`, `tomato_disease_model.keras`

## Datasets

| Dataset | Description |
|---|---|
| **Crop Recommendation** | Soil and climate parameters (N, P, K, temperature, humidity, pH, rainfall) with crop labels |
| **Crop Production** | State/district-level production records with crop, season, area, and yield data |
| **Tomato Disease** | Labeled leaf images for CNN training across 5 disease/health classes |

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/iamhriturajsaha/AGRIBOT-AI-POWERED-FARMING-ASSISTANT.git
cd AGRIBOT-AI-POWERED-FARMING-ASSISTANT
```

### 2. Create and activate a virtual environment

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux / macOS
source venv/bin/activate
```

## Usage

### Running the Streamlit App

```bash
cd "Streamlit Module"
streamlit run agribot_app.py
```

## Technologies

| Category | Tools |
|---|---|
| Language | Python 3.8+ |
| ML / DL | Scikit-Learn, TensorFlow, Keras |
| Data Processing | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| Web Interface | Streamlit |
| Model Serialization | Joblib, Pickle |

## Future Improvements

- Real-time weather API integration.
- Support for more crop types and diseases beyond tomato.
- Mobile application (Android / iOS).
- IoT sensor integration for live soil monitoring.
- Satellite imagery analysis.
- Multi-language support for regional farmers.
- Cloud deployment (AWS / GCP / Azure).

*Built to demonstrate the practical application of Machine Learning and Deep Learning in precision agriculture.*
