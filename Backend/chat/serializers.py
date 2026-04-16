from rest_framework import serializers
from .models import Chat
class ChatSerializer(serializers.ModelSerializer):
    user = serializers.StringRelatedField(read_only=True)
    class Meta:
        model = Chat
        fields = ["id", "user", "message", "response", "created_at"]
        read_only_fields = ["id", "user", "response", "created_at"]