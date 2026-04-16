# 🌾AgriBot — AI-Powered Farming Assistant

> *From soil to silicon—intelligence for every farmer.*

## 🚀 Project Overview
AgriBot is a multi-modal, AI-powered agricultural advisory system that integrates -

- 🌿 Computer Vision (Disease Detection). 
- 🌱 Machine Learning (Crop & Fertilizer Recommendation).  
- 🤖 LLM-powered Chatbot (Context-aware advisory).  
- 🌐 Interactive UI (Streamlit + React Cyberpunk Dashboard).  

It is designed as a scalable decision-support system for modern agriculture, capable of operating across different interfaces and deployment environments.

This project has three versions -
## 📊 STREAMLIT ML SYSTEM
<p align="center">
  <img src="Streamlit Version/Images/1.png" alt="1" width="1000"/><br>
  <img src="Streamlit Version/Images/2.png" alt="2" width="1000"/><br>
  <img src="Streamlit Version/Images/3.png" alt="3" width="1000"/><br>
  <img src="Streamlit Version/Images/4.png" alt="4" width="1000"/><br>
  <img src="Streamlit Version/Images/5.png" alt="5" width="1000"/><br>
</p>

## 🌌 CYBERPUNK FULL-STACK SYSTEM
<p align="center">
  <img src="Cyberpunk Version/Images/1.png" alt="1" width="1000"/><br>
  <img src="Cyberpunk Version/Images/2.png" alt="2" width="1000"/><br>
  <img src="Cyberpunk Version/Images/3.png" alt="3" width="1000"/><br>
  <img src="Cyberpunk Version/Images/4.png" alt="4" width="1000"/><br>
  <img src="Cyberpunk Version/Images/5.png" alt="5" width="1000"/><br>
  <img src="Cyberpunk Version/Images/6.png" alt="6" width="1000"/><br>
  <img src="Cyberpunk Version/Images/7.png" alt="7" width="1000"/><br>
  <img src="Cyberpunk Version/Images/8.png" alt="8" width="1000"/><br>
  <img src="Cyberpunk Version/Images/9.png" alt="9" width="1000"/><br>
  <img src="Cyberpunk Version/Images/10.png" alt="10" width="1000"/><br>
  <img src="Cyberpunk Version/Images/11.png" alt="11" width="1000"/><br>
</p>

## 🧠 CORPORATE FULL-STACK SYSTEM
<p align="center">
  <img src="Corporate Version/Images/1.png" alt="1" width="1000"/><br>
  <img src="Corporate Version/Images/2.png" alt="2" width="1000"/><br>
  <img src="Corporate Version/Images/3.png" alt="3" width="1000"/><br>
  <img src="Corporate Version/Images/4.png" alt="4" width="1000"/><br>
  <img src="Corporate Version/Images/5.png" alt="5" width="1000"/><br>
  <img src="Corporate Version/Images/6.png" alt="6" width="1000"/><br>
  <img src="Corporate Version/Images/7.png" alt="7" width="1000"/><br>
  <img src="Corporate Version/Images/8.png" alt="8" width="1000"/><br>
  <img src="Corporate Version/Images/9.png" alt="9" width="1000"/><br>
  <img src="Corporate Version/Images/10.png" alt="10" width="1000"/><br>
  <img src="Corporate Version/Images/11.png" alt="11" width="1000"/><br>
</p>

## ✨ Core Capabilities
### 🌿 1. Crop Recommendation
Predicts optimal crops using environmental and soil parameters -
- Nitrogen (N), Phosphorus (P), Potassium (K).  
- Temperature, Humidity.  
- pH, Rainfall.  
**Model -** Random Forest Classifier.  

### 🦠 2. Disease Detection (Computer Vision)
- Input - Leaf images.  
- Output - Crop + Disease classification.  
**Model -** MobileNetV2 / Custom CNN (TensorFlow/Keras).  

### 🔥 3. Explainable AI (XAI)
- Grad-CAM heatmaps highlight infected regions.  
- Provides transparency in model decisions.  

### 💊 4. Fertilizer Recommendation
Suggests optimal fertilizers based on -
- Soil composition.  
- Crop type.  
- Environmental conditions.  
**Model -** Random Forest + Label Encoding.  

### 🤖 5. Context-Aware AI Chatbot
Powered by Llama 3.1 via Groq API.

Capabilities -
- Disease explanation & treatment.  
- Fertilizer advice.  
- Crop guidance.  
- Context-aware follow-ups.  
- Context Injection (uses latest prediction automatically).

### 🧠 6. Full AI Advisory Pipeline
```
Leaf Image → Disease Detection  
Soil Data → Crop Recommendation  
Crop + Soil → Fertilizer Recommendation  
User Query → LLM Chatbot  
```
All modules combine into a unified intelligent advisory system.

## 🏗 System Architectures

### 🔹 Version 1 - Streamlit ML Pipeline
- Lightweight, end-to-end ML system.  
- Ideal for prototyping and demos.  

### 🔹 Version 2 - Cyberpunk Full-Stack System
- React + Three.js frontend.  
- Django backend with DRF.  
- Real-time AI inference.  

Features -
- Glassmorphism UI.  
- Neon cyberpunk theme.  
- Smooth animations.  

### 🔹 Version 3 - Corporate AI Platform
- Secure authentication (JWT).  
- Dashboard + history tracking.  
- Scalable API-based architecture.  

## 🧠 AI/ML Pipeline
### Disease Model
- Transfer Learning (MobileNetV2 / CNN).  
- Input - 128×128 images.  
- Output - Multi-class classification.  

### Crop Model
- RandomForestClassifier.  
- Input - Soil + environmental data.  

### Fertilizer Model
- RandomForestClassifier.  
- Requires strict feature order (`fert_columns.pkl`).  

### Chatbot
- LLM - Llama 3.1 (Groq API).  
- Domain-restricted to agriculture.  

## 🛠 Tech Stack
### Frontend
- React.js + Vite.
- Tailwind CSS.  
- Framer Motion.  
- Three.js.  
- Recharts.  

### Backend
- Django.  
- Django Rest Framework (DRF).  
- JWT Authentication (SimpleJWT).  

### AI/ML
- TensorFlow / Keras.  
- Scikit-learn.  
- NumPy. 

### Database
- SQLite (default).  
- Cloudinary / Local storage.  

## 🖥 User Interfaces
### 1. Streamlit UI
- Multi-tab interface.  
- Integrated chatbot. 
- Quick ML predictions.  

### 2. Cyberpunk UI
- 3D visualization.  
- Neon + glassmorphism.  
- Animated transitions.  

### 3. Corporate UI
- User authentication.  
- History tracking.  
- Smart insights.  

## ⚙️ Installation & Setup
### 🔹 Backend
```bash
cd backend
python manage.py runserver
```

### 🔹 Frontend
```bash
cd frontend
npm install
npm run dev
```

### 🔹 Environment Variables
```env
GROQ_API_KEY=your_groq_api_key
DJANGO_SECRET_KEY=your_secret_key
DEBUG=True
```

## ⚠️ Common Issues & Fixes

### ❌ Disease Detection Issues  
✔ Ensure dataset format - `Crop___Disease`  

### ❌ Fertilizer Model Errors  
✔ Maintain feature order using `fert_columns.pkl`  

### ❌ Model Not Found  
```python
model.save("Models/disease_model.keras")
```

### ❌ ngrok Connection Issue  
✔ Start Streamlit first and wait a few seconds  

### ❌ Chatbot Not Working  
✔ Set `GROQ_API_KEY` correctly  

## 🏆 Key Highlights
- ✅ Multi-modal AI system.  
- ✅ Deep Learning + ML integration.  
- ✅ Explainable AI (Grad-CAM).  
- ✅ Context-aware LLM chatbot.  
- ✅ Multiple UI implementations.  
- ✅ Secure & scalable architecture.  

## 🔮 Future Enhancements
- 🎤 Voice assistant (local languages).  
- 📩 SMS advisory system.  
- 📈 Market price prediction.  
- 🧠 Advanced reasoning agents.   

## 🎯 Conclusion
AgriBot evolves across three versions into a complete agricultural intelligence ecosystem -
- Version 1 → ML Prototype.  
- Version 2 → Futuristic AI System.  
- Version 3 → Production Platform.
  
By combining -
- Computer Vision  
- Machine Learning  
- LLM Intelligence  
- Modern UI/UX  

👉 AgriBot delivers scalable, intelligent and practical farming solutions.

*Built for the future of agriculture 🌱*
