from django.urls import path
from .views import DiseasePredictView, DiseaseHistoryView, BatchPredictView

urlpatterns = [
    path("predict/", DiseasePredictView.as_view(), name="disease-predict"),
    path("predict/batch/", BatchPredictView.as_view(), name="disease-batch-predict"),
    path("history/", DiseaseHistoryView.as_view(), name="disease-history"),
]