from rest_framework import status
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated
from .models import Chat
from .serializers import ChatSerializer
from openai import OpenAI
import os

client = OpenAI(
    api_key=os.getenv("GROQ_API_KEY", "no-key"),
    base_url="https://api.groq.com/openai/v1"
)

def get_bot_response(user, message, image_context=""):
    try:
        messages = [
            {
                "role": "system",
                "content": (
                    "You are the AgriBot Intelligence System, a specialized agricultural AI.\n\n"
                    "CORE DIRECTIVE: You are strictly restricted to the domain of agriculture, crop diagnostics, soil health, and farming technology.\n\n"
                    "GUARDRAILS:\n"
                    "1. ONLY process queries related to farming, plant diseases, weather, or agriculture.\n"
                    "2. DIRECTLY REJECT any questions regarding coding, politics, general knowledge, or entertainment.\n"
                    "3. If a question is off-topic, respond with: 'I am specialized in agricultural diagnostics and farming assistance. I cannot provide information on this topic. How can I help you with your crops today?'\n"
                    "4. Maintain a professional, technical, and respectful tone at all times.\n"
                )
            }
        ]

        if image_context:
            messages[0]["content"] += f"\n\nCURRENT UPLOAD_CONTEXT: {image_context}"

        # Context-Aware AI Injection
        from disease.models import DiseasePrediction
        latest_pred = DiseasePrediction.objects.filter(user=user).order_by('-created_at').first()
        if latest_pred and not image_context: # Only inject history context if they didn't just upload an image explicitly
            crop, disease = "Unknown", latest_pred.result
            if latest_pred.result and "___" in latest_pred.result:
                crop, disease = latest_pred.result.split("___")
            disease_clean = disease.replace("_", " ") if disease else "Unknown"
            
            messages[0]["content"] += (
                f"\n\nCRITICAL USER CONTEXT:\nThe user recently scanned a {crop} crop. "
                f"The AI diagnosed it with: {disease_clean} (Confidence: {round(latest_pred.confidence * 100, 1)}%). "
                f"If the user asks a vague question (e.g., 'how to treat it?'), assume they are talking about this {crop} ({disease_clean})."
            )

        prior_chats = Chat.objects.filter(user=user).order_by('-created_at')[:5]
        for chat in reversed(prior_chats):
            messages.append({"role": "user", "content": chat.message})
            if chat.response:
                messages.append({"role": "assistant", "content": chat.response})    
                
        messages.append({"role": "user", "content": message})
        
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=messages,
            temperature=0.6,
            max_tokens=300
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Chatbot failed: {e}")
        return "Sorry, I am unable to respond right now. Please try again later."

class ChatView(APIView):
    permission_classes = [IsAuthenticated]

    def post(self, request):
        msg = request.data.get("message", "")
        if not str(msg).strip():
            msg = "Here is an image I uploaded. Can you tell me what disease it has and how to treat it?"

        user = request.user
        image_context = ""
        image = request.FILES.get("image")

        if image:
            try:
                from disease.views import predict_image
                crop, disease, conf = predict_image(image)
                disease_clean = disease.replace("_", " ") if disease else "Unknown condition"
                image_context = f"[The user attached a photo of a {crop}. The AgriBot vision model diagnosed it as: {disease_clean} with {round(conf*100, 1)}% confidence.]"
            except Exception as e:
                print(f"Vision API Error: {e}")
                image_context = "[The user attached an image, but the vision model failed to analyze it.]"

        bot_response = get_bot_response(user, msg, image_context)
        
        db_msg = msg
        if image_context:
            db_msg += f"\n{image_context}"
            
        chat = Chat.objects.create(user=user, message=db_msg, response=bot_response)

        return Response(
            {
                "status": "success",
                "message": "Chat processed successfully",
                "data": ChatSerializer(chat).data
            },
            status=status.HTTP_201_CREATED
        )

class ChatClearView(APIView):
    permission_classes = [IsAuthenticated]

    def delete(self, request):
        Chat.objects.filter(user=request.user).delete()
        return Response({"status": "success", "message": "Chat history cleared"}, status=status.HTTP_200_OK)