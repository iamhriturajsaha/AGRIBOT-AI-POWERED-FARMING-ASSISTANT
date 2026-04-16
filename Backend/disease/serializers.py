from rest_framework import serializers
from .models import DiseasePrediction
class DiseasePredictionSerializer(serializers.ModelSerializer):
    user = serializers.StringRelatedField(read_only=True)
    class Meta:
        model = DiseasePrediction
        fields = ["id", "user", "image", "result", "confidence", "created_at"]
        read_only_fields = ["id", "user", "result", "confidence", "created_at"]