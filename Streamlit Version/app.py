import streamlit as st
import pickle
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from openai import OpenAI
import os
st.set_page_config(
    page_title="AgriBOT AI",
    page_icon="🌾",
    layout="wide"
)
st.markdown(f"""
<style>
.stApp {{
    background: url("https://images.unsplash.com/photo-1711397651462-3b2a22f5cfc8?q=80&w=1470&auto=format&fit=crop") no-repeat center center fixed;
    background-size: cover;
}}

.block-container {{
    background: rgba(0, 0, 0, 0.55);
    padding: 2rem;
    border-radius: 20px;
    backdrop-filter: blur(10px);
}}
h1, h2, h3, h4, h5 {{
    color: #ffffff;
}}
.stMarkdown, label {{
    color: #e6e6e6 !important;
}}
.stButton>button {{
    border-radius: 10px;
    background: linear-gradient(90deg, #00c853, #64dd17);
    color: white;
    font-weight: bold;
}}
.stTabs [data-baseweb="tab"] {{
    font-size: 18px;
}}
</style>
""", unsafe_allow_html=True)
st.markdown("""
# 🌾 AgriBOT — AI Farming Assistant
### 🌱 Smart Agriculture using AI
""")
MODEL_DIR = "Models"
crop_model = pickle.load(open(os.path.join(MODEL_DIR, "crop_model.pkl"), "rb"))
fert_model = pickle.load(open(os.path.join(MODEL_DIR, "fert_model.pkl"), "rb"))
encoders = pickle.load(open(os.path.join(MODEL_DIR, "fert_encoders.pkl"), "rb"))
fert_columns = pickle.load(open(os.path.join(MODEL_DIR, "fert_columns.pkl"), "rb"))
class_indices = pickle.load(open(os.path.join(MODEL_DIR, "class_indices.pkl"), "rb"))
class_labels = list(class_indices.keys()) 
model = load_model(
    os.path.join(MODEL_DIR, "disease_model.keras"),
    compile=False
)
client = OpenAI(
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1"
)
def is_agri_query(query):
    keywords = ["crop","plant","disease","fertilizer","soil","farming"]
    return any(word in query.lower() for word in keywords)
def agri_chatbot(query):
    if not is_agri_query(query):
        return "I can only help with agriculture-related questions."
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system","content":"You are an agriculture expert."},
            {"role": "user","content":query}
        ]
    )
    return response.choices[0].message.content
def prepare_fert_input(user_input):
    data = []
    for col in fert_columns:
        val = user_input.get(col, 0)
        if col in encoders:
            try:
                val = encoders[col].transform([val])[0]
            except:
                val = 0
        data.append(float(val))
    return data
def predict_disease(img_file):
    img = Image.open(img_file).convert("RGB")
    img = img.resize((128,128))
    img = np.array(img)/255.0
    img = np.expand_dims(img, axis=0)
    pred = model.predict(img)
    label = class_labels[np.argmax(pred)]
    return label.split("___")
tab1, tab2, tab3, tab4 = st.tabs([
    "🌿 Crop",
    "🦠 Disease",
    "🤖 AI Advisor",
    "💬 Chatbot"
])
with tab1:
    st.subheader("🌿 Crop Recommendation")
    col1, col2 = st.columns(2)
    with col1:
        N = st.slider("Nitrogen",0,140,50)
        P = st.slider("Phosphorus",0,140,50)
        K = st.slider("Potassium",0,140,50)
    with col2:
        temp = st.slider("Temperature",0,50,25)
        humidity = st.slider("Humidity",0,100,60)
        ph = st.slider("pH",0.0,14.0,6.5)
        rainfall = st.slider("Rainfall",0,300,100)
    if st.button("🌱 Recommend Crop"):
        crop = crop_model.predict([[N,P,K,temp,humidity,ph,rainfall]])[0]
        st.success(f"Recommended Crop: {crop}")
with tab2:
    st.subheader("🦠 Disease Detection")
    file = st.file_uploader("Upload Leaf Image", type=["jpg","png"])
    if file:
        st.image(file, use_container_width=True)
        with st.spinner("Analyzing..."):
            crop_img, disease = predict_disease(file)
        st.success(f"Crop: {crop_img}")
        st.success(f"Disease: {disease}")
with tab3:
    st.subheader("🤖 Full AI Advisor")
    col1, col2 = st.columns(2)
    with col1:
        file = st.file_uploader("Upload Image", key="full")
    with col2:
        N = st.slider("N",0,140,50, key="n2")
        P = st.slider("P",0,140,50, key="p2")
        K = st.slider("K",0,140,50, key="k2")
        temp = st.slider("Temp",0,50,25, key="t2")
        humidity = st.slider("Humidity",0,100,60, key="h2")
        ph = st.slider("pH",0.0,14.0,6.5, key="ph2")
        rainfall = st.slider("Rainfall",0,300,100, key="r2")
    if st.button("🚀 Run AI"):
        if file:
            crop_img, disease = predict_disease(file)
            crop = crop_model.predict([[N,P,K,temp,humidity,ph,rainfall]])[0]
            fert_input = {
                "Temperature": temp,
                "Humidity": humidity,
                "Soil_pH": ph,
                "Nitrogen_Level": N,
                "Phosphorus_Level": P,
                "Potassium_Level": K,
                "Crop_Type": crop
            }
            fert = fert_model.predict([prepare_fert_input(fert_input)])[0]
            st.success(f"""
🌿 Detected Crop: {crop_img}
🦠 Disease: {disease}
🌱 Recommended Crop: {crop}
💊 Fertilizer: {fert}
""")
with tab4:
    st.subheader("💬 AI Chatbot")
    query = st.text_input("Ask something about farming...")
    if query:
        with st.spinner("Thinking..."):
            response = agri_chatbot(query)
        st.success(response)
st.markdown("---")
st.markdown("🚀 Built with AI")
