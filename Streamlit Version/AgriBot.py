# INSTALL LIBRARIES
!pip install -q tensorflow scikit-learn pillow streamlit pyngrok uvicorn nest-asyncio openai opencv-python

# IMPORTS
import os, zipfile, shutil, pickle
import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.callbacks import *
from sklearn.utils.class_weight import compute_class_weight

# DATASET PREPARATION
with zipfile.ZipFile("Crops.zip", 'r') as zip_ref:
    zip_ref.extractall("Crops")
source_dir = "Crops"
target_dir = "Processed"
if os.path.exists(target_dir):
    shutil.rmtree(target_dir)
os.makedirs(target_dir)
for split in os.listdir(source_dir):
    for crop in os.listdir(os.path.join(source_dir, split)):
        crop_path = os.path.join(source_dir, split, crop)
        for inner in os.listdir(crop_path):
            for disease in os.listdir(os.path.join(crop_path, inner)):
                disease_path = os.path.join(crop_path, inner, disease)
                new_folder = f"{crop}___{disease}"
                new_path = os.path.join(target_dir, new_folder)
                os.makedirs(new_path, exist_ok=True)
                for img in os.listdir(disease_path):
                    shutil.copy(os.path.join(disease_path, img), new_path)
print("✅ Dataset Ready")

# VERIFY DATASET
print(os.listdir("Processed")[:10])

# DATA GENERATOR
img_size = (128,128)
batch_size = 32
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    validation_split=0.2,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True,
    shear_range=0.1
)
val_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    validation_split=0.2
)
train_data = train_datagen.flow_from_directory(
    "Processed",
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='training'
)
val_data = val_datagen.flow_from_directory(
    "Processed",
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation'
)

# CLASS WEIGHTS
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(train_data.classes),
    y=train_data.classes
)
class_weights = dict(enumerate(class_weights))
print(class_weights)

# CROP MODEL
crop_df = pd.read_csv("Crop Recommendation.csv")
X_crop = crop_df.drop("label", axis=1)
y_crop = crop_df["label"]
crop_model = RandomForestClassifier()
crop_model.fit(X_crop, y_crop)

# FERTILIZER MODEL
fert_df = pd.read_csv("Fertilizer Recommendation.csv")
target_col = "Recommended_Fertilizer"
X_fert = fert_df.drop(target_col, axis=1)
y_fert = fert_df[target_col]
encoders = {}
for col in X_fert.columns:
    if X_fert[col].dtype == 'object':
        le = LabelEncoder()
        X_fert[col] = le.fit_transform(X_fert[col])
        encoders[col] = le
fert_model = RandomForestClassifier()
fert_model.fit(X_fert, y_fert)

# DISEASE MODEL
# MobileNet
base_model = MobileNetV2(
    weights='imagenet',
    include_top=False,
    input_shape=(128,128,3)
)
base_model.trainable = False
inputs = Input(shape=(128,128,3))
x = base_model(inputs, training=False)
x = GlobalAveragePooling2D()(x)
x = BatchNormalization()(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.5)(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.3)(x)
outputs = Dense(train_data.num_classes, activation='softmax')(x)
model = Model(inputs, outputs)
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
    metrics=['accuracy']
)

# Train MobileNet
callbacks = [
    EarlyStopping(patience=30, restore_best_weights=True),
    ReduceLROnPlateau(patience=2, factor=0.3)
]
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=30,
    class_weight=class_weights,
    callbacks=callbacks
)

# Fine Tuning
for layer in base_model.layers[-50:]:
    layer.trainable = True
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
    metrics=['accuracy']
)
history_fine = model.fit(
    train_data,
    validation_data=val_data,
    epochs=30,
    class_weight=class_weights,
    callbacks=callbacks
)

# EVALUATION
loss, acc = model.evaluate(val_data)
print("Accuracy:", acc)

# SAVING MODELS
SAVE_DIR = "Models"
os.makedirs(SAVE_DIR, exist_ok=True)
with open(os.path.join(SAVE_DIR, "crop_model.pkl"), "wb") as f:
    pickle.dump(crop_model, f)
with open(os.path.join(SAVE_DIR, "fert_model.pkl"), "wb") as f:
    pickle.dump(fert_model, f)
with open(os.path.join(SAVE_DIR, "fert_encoders.pkl"), "wb") as f:
    pickle.dump(encoders, f)
with open(os.path.join(SAVE_DIR, "fert_columns.pkl"), "wb") as f:
    pickle.dump(X_fert.columns.tolist(), f)
model.save(os.path.join(SAVE_DIR, "disease_model.keras"))
class_indices = train_data.class_indices
with open(os.path.join(SAVE_DIR, "class_indices.pkl"), "wb") as f:
    pickle.dump(class_indices, f)
print("✅ All models saved successfully")

# ENSEMBLE PREDICTION
with open("Models/class_indices.pkl", "rb") as f:
    class_indices = pickle.load(f)
class_labels = list(class_indices.keys())
def preprocess_img(img_file):
    img = Image.open(img_file).convert("RGB")
    img = img.resize((128,128))
    img = np.array(img)
    img = preprocess_input(img)
    return np.expand_dims(img, axis=0)
def predict_disease(img_file):
    img = preprocess_img(img_file)
    preds = model.predict(img, verbose=0)
    pred_index = np.argmax(preds)
    confidence = float(np.max(preds))
    label = class_labels[pred_index]
    crop, disease = label.split("___")
    return {
        "crop": crop,
        "disease": disease,
        "confidence": round(confidence, 4)
    }
result = predict_disease("Crop.JPG")
print(result)

# CHATBOT
from openai import OpenAI
import os
client = OpenAI(
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1"
)
def agri_chatbot(query):
    try:
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an expert agricultural assistant.\n\n"
                        "Rules:\n"
                        "1. Answer ONLY agriculture-related questions.\n"
                        "2. Agriculture includes crops, diseases, fertilizers, soil, farming.\n"
                        "3. If the question is NOT related, reply EXACTLY:\n"
                        "'I can only help with agriculture-related questions like crops, diseases, and fertilizers.'\n\n"
                        "4. Be helpful and practical."
                    )
                },
                {"role": "user", "content": query}
            ],
            temperature=0.5,
            max_tokens=250
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Error: {str(e)}"

# Commented out IPython magic to ensure Python compatibility.
# # STREAMLIT UI
# %%writefile app.py
# import streamlit as st
# import pickle
# import numpy as np
# from PIL import Image
# from tensorflow.keras.models import load_model
# from openai import OpenAI
# import os
# st.set_page_config(
#     page_title="AgriBOT AI",
#     page_icon="🌾",
#     layout="wide"
# )
# st.markdown(f"""
# <style>
# .stApp {{
#     background: url("https://images.unsplash.com/photo-1711397651462-3b2a22f5cfc8?q=80&w=1470&auto=format&fit=crop") no-repeat center center fixed;
#     background-size: cover;
# }}
# 
# .block-container {{
#     background: rgba(0, 0, 0, 0.55);
#     padding: 2rem;
#     border-radius: 20px;
#     backdrop-filter: blur(10px);
# }}
# h1, h2, h3, h4, h5 {{
#     color: #ffffff;
# }}
# .stMarkdown, label {{
#     color: #e6e6e6 !important;
# }}
# .stButton>button {{
#     border-radius: 10px;
#     background: linear-gradient(90deg, #00c853, #64dd17);
#     color: white;
#     font-weight: bold;
# }}
# .stTabs [data-baseweb="tab"] {{
#     font-size: 18px;
# }}
# </style>
# """, unsafe_allow_html=True)
# st.markdown("""
# # 🌾 AgriBOT — AI Farming Assistant
# ### 🌱 Smart Agriculture using AI
# """)
# MODEL_DIR = "Models"
# crop_model = pickle.load(open(os.path.join(MODEL_DIR, "crop_model.pkl"), "rb"))
# fert_model = pickle.load(open(os.path.join(MODEL_DIR, "fert_model.pkl"), "rb"))
# encoders = pickle.load(open(os.path.join(MODEL_DIR, "fert_encoders.pkl"), "rb"))
# fert_columns = pickle.load(open(os.path.join(MODEL_DIR, "fert_columns.pkl"), "rb"))
# class_indices = pickle.load(open(os.path.join(MODEL_DIR, "class_indices.pkl"), "rb"))
# class_labels = list(class_indices.keys())
# model = load_model(
#     os.path.join(MODEL_DIR, "disease_model.keras"),
#     compile=False
# )
# client = OpenAI(
#     api_key=os.getenv("GROQ_API_KEY"),
#     base_url="https://api.groq.com/openai/v1"
# )
# def is_agri_query(query):
#     keywords = ["crop","plant","disease","fertilizer","soil","farming"]
#     return any(word in query.lower() for word in keywords)
# def agri_chatbot(query):
#     if not is_agri_query(query):
#         return "I can only help with agriculture-related questions."
#     response = client.chat.completions.create(
#         model="llama-3.1-8b-instant",
#         messages=[
#             {"role": "system","content":"You are an agriculture expert."},
#             {"role": "user","content":query}
#         ]
#     )
#     return response.choices[0].message.content
# def prepare_fert_input(user_input):
#     data = []
#     for col in fert_columns:
#         val = user_input.get(col, 0)
#         if col in encoders:
#             try:
#                 val = encoders[col].transform([val])[0]
#             except:
#                 val = 0
#         data.append(float(val))
#     return data
# def predict_disease(img_file):
#     img = Image.open(img_file).convert("RGB")
#     img = img.resize((128,128))
#     img = np.array(img)/255.0
#     img = np.expand_dims(img, axis=0)
#     pred = model.predict(img)
#     label = class_labels[np.argmax(pred)]
#     return label.split("___")
# tab1, tab2, tab3, tab4 = st.tabs([
#     "🌿 Crop",
#     "🦠 Disease",
#     "🤖 AI Advisor",
#     "💬 Chatbot"
# ])
# with tab1:
#     st.subheader("🌿 Crop Recommendation")
#     col1, col2 = st.columns(2)
#     with col1:
#         N = st.slider("Nitrogen",0,140,50)
#         P = st.slider("Phosphorus",0,140,50)
#         K = st.slider("Potassium",0,140,50)
#     with col2:
#         temp = st.slider("Temperature",0,50,25)
#         humidity = st.slider("Humidity",0,100,60)
#         ph = st.slider("pH",0.0,14.0,6.5)
#         rainfall = st.slider("Rainfall",0,300,100)
#     if st.button("🌱 Recommend Crop"):
#         crop = crop_model.predict([[N,P,K,temp,humidity,ph,rainfall]])[0]
#         st.success(f"Recommended Crop: {crop}")
# with tab2:
#     st.subheader("🦠 Disease Detection")
#     file = st.file_uploader("Upload Leaf Image", type=["jpg","png"])
#     if file:
#         st.image(file, use_container_width=True)
#         with st.spinner("Analyzing..."):
#             crop_img, disease = predict_disease(file)
#         st.success(f"Crop: {crop_img}")
#         st.success(f"Disease: {disease}")
# with tab3:
#     st.subheader("🤖 Full AI Advisor")
#     col1, col2 = st.columns(2)
#     with col1:
#         file = st.file_uploader("Upload Image", key="full")
#     with col2:
#         N = st.slider("N",0,140,50, key="n2")
#         P = st.slider("P",0,140,50, key="p2")
#         K = st.slider("K",0,140,50, key="k2")
#         temp = st.slider("Temp",0,50,25, key="t2")
#         humidity = st.slider("Humidity",0,100,60, key="h2")
#         ph = st.slider("pH",0.0,14.0,6.5, key="ph2")
#         rainfall = st.slider("Rainfall",0,300,100, key="r2")
#     if st.button("🚀 Run AI"):
#         if file:
#             crop_img, disease = predict_disease(file)
#             crop = crop_model.predict([[N,P,K,temp,humidity,ph,rainfall]])[0]
#             fert_input = {
#                 "Temperature": temp,
#                 "Humidity": humidity,
#                 "Soil_pH": ph,
#                 "Nitrogen_Level": N,
#                 "Phosphorus_Level": P,
#                 "Potassium_Level": K,
#                 "Crop_Type": crop
#             }
#             fert = fert_model.predict([prepare_fert_input(fert_input)])[0]
#             st.success(f"""
# 🌿 Detected Crop: {crop_img}
# 🦠 Disease: {disease}
# 🌱 Recommended Crop: {crop}
# 💊 Fertilizer: {fert}
# """)
# with tab4:
#     st.subheader("💬 AI Chatbot")
#     query = st.text_input("Ask something about farming...")
#     if query:
#         with st.spinner("Thinking..."):
#             response = agri_chatbot(query)
#         st.success(response)
# st.markdown("---")
# st.markdown("🚀 Built with AI")

# RUN STREAMLIT
import time
import subprocess
from pyngrok import ngrok, conf
NGROK_AUTH_TOKEN = "2z0Oqv0tD166fELGCHwV2gLZwq1_2G2zUQRSs6C27k9vdzxwq"
conf.get_default().auth_token = NGROK_AUTH_TOKEN
!pkill -f streamlit || echo "No running Streamlit"
app_file = "app.py"
process = subprocess.Popen(
    ["streamlit", "run", app_file, "--server.port", "8501", "--server.address", "0.0.0.0"],
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE
)
time.sleep(8)
public_url = ngrok.connect(8501)
print("🚀 Your Streamlit App is LIVE at:")
print(public_url)
