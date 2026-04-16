# Corporate Version

## ✨ Key Features
### 🔍 1. AI Disease Diagnosis
Upload photos of crops to receive instant diagnostic reports. AgriBot uses a custom-trained TensorFlow model to identify diseases across dozens of plant species with high confidence levels.

### 🤖 2. Context-Aware AI Copilot
Powered by Llama 3.1 (Groq Cloud), the AgriBot assistant is more than just a chatbot -
- **Domain-Locked** - Strictly restricted to agricultural topics to ensure reliability.
- **Contextual Intelligence** - Automatically retrieves the user's latest diagnostic history. If you ask "How do I treat it?", the bot knows exactly which crop and disease you are referring to from your previous scan.

### 📊 3. Smart Dashboard
A premium, dark-mode dashboard featuring glassmorphism design that simplifies farm management, history tracking and diagnostic overviews.

### 🔐 4. Secure Authentication
Full JWT-based authentication system ensuring user data and diagnostic history are protected and personalized.

## 🛠 Tech Stack
| Layer | Technologies |
| :--- | :--- |
| **Frontend** | React.js, Vite, Tailwind CSS, Framer Motion, Lucide Icons |
| **Backend** | Django, Django Rest Framework (DRF), SimpleJWT |
| **AI/ML** | TensorFlow, Keras, Groq Cloud API (Llama 3.1), NumPy |
| **Database** | SQLite (Default), Cloudinary (for media hosting/local storage) |

## 🧠 Deep Dive - The AgriBot Logic
### Contextual Chat Injection
The AgriBot Assistant uses a unique "Context Injection" pattern. When a user sends a message, the system performs a lookup -
1. It fetches the most recent `DiseasePrediction` for that user.
2. It injects this data (Crop Name, Disease, Confidence) into the System Prompt.
3. This allows the LLM to provide relevant advice without the user having to repeat details about their crop scan.

### Domain Filtering
The assistance is guarded by strict system instructions -
> *"You are bound EXCLUSIVELY to agriculture, farming, crops and agronomy. If the user asks ANY question outside of agriculture, you MUST immediately reject it."*
