from rest_framework import status
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated
from PIL import Image
import numpy as np
import os
import pickle
from openai import OpenAI
from django.conf import settings
from .models import DiseasePrediction
from .serializers import DiseasePredictionSerializer
from .ml_model import load_model
def load_class_labels():
    class_path = os.path.join(os.path.dirname(__file__), "class_indices.pkl")
    if not os.path.exists(class_path):
        raise FileNotFoundError(f"{class_path} not found")
    with open(class_path, "rb") as f:
        class_indices = pickle.load(f)
    return {v: k for k, v in class_indices.items()}
class_labels = load_class_labels()
import json
client = OpenAI(
    api_key=os.getenv("GROQ_API_KEY", "no_key"),
    base_url="https://api.groq.com/openai/v1"
)

def get_treatment(crop, disease):
    d = disease.lower()
    
    # Try dynamic AI Insight (Google Magic)
    try:
        if os.getenv("GROQ_API_KEY"):
            if "healthy" in d:
                prompt = f"The crop {crop} was diagnosed as healthy. Provide a JSON response with cause (Optimal conditions), action (How to maintain), and risk_level (None)."
            else:
                prompt = f"The crop {crop} was diagnosed with {disease}. Provide a JSON response with 'cause' (brief explanation of why this disease happens), 'action' (specific treatment or fertilizer), and 'risk_level' (High, Medium, Critical)."
            
            response = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[
                    {"role": "system", "content": "You are an expert plant pathologist. Always return ONLY valid JSON with keys: 'cause', 'action', 'risk_level'. Do not include markdown blocks or any other text."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=200
            )
            content = response.choices[0].message.content.strip()
            if content.startswith("```json"):
                content = content[7:-3].strip()
            elif content.startswith("```"):
                content = content[3:-3].strip()
                
            data = json.loads(content)
            return {
                "cause": data.get("cause", "Optimal conditions." if "healthy" in d else "Unknown pathogen."),
                "action": data.get("action", "Maintain routine." if "healthy" in d else "Consult a specialist."),
                "risk_level": data.get("risk_level", "None" if "healthy" in d else "Unknown")
            }
    except Exception as e:
        print(f"AI Insight Generation failed: {e}")

    # Fallback to rules if API fails or no key
    if "healthy" in d:
        return {
            "cause": "Optimal conditions.",
            "action": "Maintain current watering and nutrition routines.",
            "risk_level": "None"
        }
    elif "blight" in d:
        return {
            "cause": "Fungal infection often triggered by high humidity and poor air circulation.",
            "action": "Apply copper-based fungicide, remove infected plant parts, and avoid overhead watering.",
            "risk_level": "High"
        }
    elif "rot" in d:
        return {
            "cause": "Soil-borne pathogens or excessive watering causing root/stem decay.",
            "action": "Improve soil drainage, apply targeted fungicide, and discard heavily infected plants.",
            "risk_level": "Critical"
        }
    elif "spot" in d:
        return {
            "cause": "Bacterial or fungal spores spreading via water splash.",
            "action": "Apply appropriate bacterial/fungal sprays and prune affected leaves.",
            "risk_level": "Medium"
        }
    return {
        "cause": "Unknown pathogen or environmental stress.",
        "action": "Isolate the plant and consult an agricultural expert.",
        "risk_level": "Unknown"
    }
def preprocess_image(image_file):
    try:
        img = Image.open(image_file).convert("RGB")
        img = img.resize((128, 128))
        img_array = np.array(img, dtype=np.float32) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        return img_array
    except Exception:
        raise ValueError("Invalid image file")
def predict_image(image_file):
    img_array = preprocess_image(image_file)
    model = load_model()
    prediction = model.predict(img_array, verbose=0)
    predicted_index = int(np.argmax(prediction[0]))
    confidence = float(np.max(prediction[0]))
    label = class_labels.get(predicted_index, f"Class_{predicted_index}")
    if "___" in label:
        crop, disease = label.split("___")
    else:
        crop, disease = label, "Unknown"
    return crop, disease, round(confidence, 4)
class DiseasePredictView(APIView):
    permission_classes = [IsAuthenticated]
    def post(self, request):
        image = request.FILES.get("image")
        if not image:
            return Response(
                {"status": "error", "message": "No image uploaded"},
                status=status.HTTP_400_BAD_REQUEST
            )
        if not image.name.lower().endswith((".jpg", ".jpeg", ".png")):
            return Response(
                {"status": "error", "message": "Invalid image format"},
                status=status.HTTP_400_BAD_REQUEST
            )
        try:
            crop, disease, confidence = predict_image(image)
            image.seek(0)
            if confidence < 0.0:
                return Response(
                    {
                        "status": "error",
                        "message": "Low confidence prediction. Try a clearer image."
                    },
                    status=status.HTTP_400_BAD_REQUEST
                )
        except ValueError as e:
            return Response(
                {"status": "error", "message": str(e)},
                status=status.HTTP_400_BAD_REQUEST
            )
        except Exception as e:
            return Response(
                {
                    "status": "error",
                    "message": "Prediction failed",
                    "details": str(e)
                },
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )
        disease_clean = disease.replace("_", " ")
        prediction_text = f"{crop} {disease_clean}"
        serializer = DiseasePredictionSerializer(data={"image": image})
        if serializer.is_valid():
            record = serializer.save(
                user=request.user,
                result=f"{crop}___{disease}",
                confidence=confidence
            )
            
            heatmap_url = None
            try:
                from .ml_model import make_gradcam_heatmap, save_and_display_gradcam
                model = load_model()
                img_array = preprocess_image(record.image.path)
                heatmap = make_gradcam_heatmap(img_array, model)
                if heatmap is not None:
                    cam_filename = f"cam_{record.id}.jpg"
                    cam_path = os.path.join(os.path.dirname(record.image.path), cam_filename)
                    save_and_display_gradcam(record.image.path, heatmap, cam_path)
                    heatmap_url = request.build_absolute_uri(f"{settings.MEDIA_URL}disease_images/{cam_filename}")
            except Exception as e:
                print(f"Grad-CAM Failed: {e}")

            return Response(
                {
                    "status": "success",
                    "message": "Prediction successful",
                    "data": {
                        "prediction": prediction_text,
                        "crop": crop,
                        "disease": disease_clean,
                        "confidence": confidence,
                        "is_healthy": "healthy" in disease.lower(),
                        "treatment": get_treatment(crop, disease),
                        "record_id": record.id,
                        "heatmap_url": heatmap_url
                    }
                },
                status=status.HTTP_201_CREATED
            )
        return Response(
            {
                "status": "error",
                "errors": serializer.errors
            },
            status=status.HTTP_400_BAD_REQUEST
        )

class BatchPredictView(APIView):
    permission_classes = [IsAuthenticated]
    
    def post(self, request):
        images = request.FILES.getlist("images")
        if not images:
            return Response({"status": "error", "message": "No images uploaded"}, status=status.HTTP_400_BAD_REQUEST)
        
        results = []
        for image in images:
            try:
                crop, disease, confidence = predict_image(image)
                image.seek(0)
                disease_clean = disease.replace("_", " ")
                prediction_text = f"{crop} {disease_clean}"
                
                serializer = DiseasePredictionSerializer(data={"image": image})
                if serializer.is_valid():
                    record = serializer.save(
                        user=request.user,
                        result=f"{crop}___{disease}",
                        confidence=confidence
                    )
                    
                    heatmap_url = None
                    try:
                        from .ml_model import make_gradcam_heatmap, save_and_display_gradcam, load_model
                        model = load_model()
                        img_array = preprocess_image(record.image.path)
                        heatmap = make_gradcam_heatmap(img_array, model)
                        if heatmap is not None:
                            cam_filename = f"cam_{record.id}.jpg"
                            cam_path = os.path.join(os.path.dirname(record.image.path), cam_filename)
                            save_and_display_gradcam(record.image.path, heatmap, cam_path)
                            heatmap_url = request.build_absolute_uri(f"{settings.MEDIA_URL}disease_images/{cam_filename}")
                    except Exception as e:
                        print(f"Batch Grad-CAM Failed: {e}")

                    results.append({
                        "filename": image.name,
                        "prediction": prediction_text,
                        "crop": crop,
                        "disease": disease_clean,
                        "confidence": confidence,
                        "is_healthy": "healthy" in disease.lower(),
                        "heatmap_url": heatmap_url,
                        "treatment": get_treatment(crop, disease)
                    })
            except Exception as e:
                results.append({
                    "filename": image.name,
                    "error": str(e)
                })
                
        return Response({
            "status": "success",
            "message": f"Processed {len(images)} images",
            "data": results
        }, status=status.HTTP_201_CREATED)

class DiseaseHistoryView(APIView):
    permission_classes = [IsAuthenticated]
    
    def get(self, request):
        history = DiseasePrediction.objects.filter(user=request.user).order_by('-created_at')
        serializer = DiseasePredictionSerializer(history, many=True, context={'request': request})
        return Response({
            "status": "success",
            "data": serializer.data
        }, status=status.HTTP_200_OK)