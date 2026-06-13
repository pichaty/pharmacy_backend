from django.db import models
import uuid

class ChatSession(models.Model):
    """เก็บแต่ละ session การพูดคุย"""
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    title = models.CharField(max_length=255, blank=True, default="เคสใหม่")

    class Meta:
        ordering = ['-updated_at']

    def __str__(self):
        return f"{self.title} ({self.created_at.strftime('%d/%m/%Y %H:%M')})"


class ChatMessage(models.Model):
    """เก็บแต่ละข้อความใน session"""
    ROLE_CHOICES = [
        ('user', 'เภสัชกร'),
        ('assistant', 'ผู้ช่วย'),
    ]
    session = models.ForeignKey(
        ChatSession,
        on_delete=models.CASCADE,
        related_name='messages'
    )
    role = models.CharField(max_length=10, choices=ROLE_CHOICES)
    content = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['created_at']

    def __str__(self):
        return f"{self.role}: {self.content[:50]}"