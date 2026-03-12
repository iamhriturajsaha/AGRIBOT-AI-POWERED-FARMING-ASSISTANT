import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
import joblib
from PIL import Image
st.set_page_config(
    page_title="AgriBot AI Farming Assistant",
    page_icon="🌱",
    layout="wide"
)

# LOAD MODELS
@st.cache_resource
def load_models():
    crop_model = joblib.load("crop_recommendation_model.pkl")
    scaler = joblib.load("crop_scaler.pkl")
    label_encoder = joblib.load("crop_label_encoder.pkl")
    yield_model = joblib.load("yield_prediction_model.pkl")
    feature_columns = joblib.load("yield_feature_columns.pkl")
    disease_model = tf.keras.models.load_model("tomato_disease_model.keras")
    return crop_model, scaler, label_encoder, yield_model, feature_columns, disease_model
crop_model, scaler, label_encoder, yield_model, feature_columns, disease_model = load_models()

# DISEASE CLASSES
disease_classes = [
"Tomato Bacterial Spot",
"Tomato Early Blight",
"Tomato Healthy",
"Tomato Late Blight",
"Tomato Leaf Mold",
"Tomato Mosaic Virus",
"Tomato Septoria Leaf Spot",
"Tomato Spider Mites",
"Tomato Target Spot",
"Tomato Yellow Leaf Curl Virus"
]

# SIDEBAR NAVIGATION
st.sidebar.title("🌱 AgriBot")
page = st.sidebar.radio(
    "Navigate",
    [
        "Home",
        "Crop Recommendation",
        "Yield Prediction",
        "Disease Detection"
    ]
)

# HOME PAGE
if page == "Home":
    st.title("🌾 AgriBot: AI-Powered Farming Assistant")
    st.markdown("""
    AgriBot helps farmers make smarter decisions using **Artificial Intelligence**.

    Features included:

    • 🌱 Crop Recommendation  
    • 📈 Crop Yield Prediction  
    • 🦠 Tomato Disease Detection  

    Built using **Machine Learning + Deep Learning**.
    """)
    st.image(
        "https://images.unsplash.com/photo-1500382017468-9049fed747ef",
        use_container_width=True
    )

# CROP RECOMMENDATION
elif page == "Crop Recommendation":
    st.title("🌱 Smart Crop Recommendation")
    col1, col2 = st.columns(2)
    with col1:
        N = st.number_input("Nitrogen (N)", 0, 200)
        P = st.number_input("Phosphorus (P)", 0, 200)
        K = st.number_input("Potassium (K)", 0, 200)
        temperature = st.number_input("Temperature (°C)", 0.0, 50.0)
    with col2:
        humidity = st.number_input("Humidity (%)", 0.0, 100.0)
        ph = st.number_input("Soil pH", 0.0, 14.0)
        rainfall = st.number_input("Rainfall (mm)", 0.0, 500.0)
    if st.button("Recommend Crop"):
        input_data = np.array([[N,P,K,temperature,humidity,ph,rainfall]])
        input_scaled = scaler.transform(input_data)
        prediction = crop_model.predict(input_scaled)
        crop = label_encoder.inverse_transform(prediction)
        st.success(f"Recommended Crop: **{crop[0]}**")

# YIELD PREDICTION
elif page == "Yield Prediction":
    st.title("📈 Crop Yield Prediction")
    state = st.text_input("State")
    crop = st.text_input("Crop")
    season = st.text_input("Season")
    area = st.number_input("Area (hectares)",0.0)
    if st.button("Predict Yield"):
        input_df = pd.DataFrame({
            "Crop_Year":[2020],
            "State_Name":[state],
            "Season":[season],
            "Crop":[crop],
            "Area":[area]
        })
        input_encoded = pd.get_dummies(input_df)

        # Align features with training data
        input_encoded = input_encoded.reindex(columns=feature_columns, fill_value=0)
        prediction = yield_model.predict(input_encoded)
        st.success(f"Predicted Yield: {prediction[0]:.2f}")

# DISEASE DETECTION
elif page == "Disease Detection":
    st.title("🦠 Tomato Disease Detection")
    uploaded_file = st.file_uploader("Upload Tomato Leaf Image")
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)
        img = image.resize((224,224))
        img = np.array(img)/255.0
        img = np.expand_dims(img, axis=0)
        prediction = disease_model.predict(img)
        predicted_class = disease_classes[np.argmax(prediction)]
        confidence = np.max(prediction)
        st.success(f"Disease: **{predicted_class}**")
        st.write(f"Confidence: {confidence:.2f}")
