from django.db import models
from users.models import User
class Chat(models.Model):
    user = models.ForeignKey(
        User,
        on_delete=models.CASCADE,
        related_name="chats",
        null=True,
        blank=True
    )
    message = models.TextField()
    response = models.TextField(blank=True, null=True)
    created_at = models.DateTimeField(auto_now_add=True)
    def __str__(self):
        username = self.user.username if self.user else "Anonymous"
        return f"{username}: {self.message[:50]}"
    class Meta:
        ordering = ["-created_at"]