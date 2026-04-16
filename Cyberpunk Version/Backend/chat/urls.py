from django.urls import path
from .views import ChatView, ChatClearView
urlpatterns = [
    path("message/", ChatView.as_view(), name="chat-message"),
    path("clear/", ChatClearView.as_view(), name="chat-clear"),
]