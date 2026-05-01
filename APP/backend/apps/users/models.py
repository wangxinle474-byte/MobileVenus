from django.db import models


class MiniProgramUser(models.Model):
    client_id = models.CharField(max_length=64, unique=True)
    openid = models.CharField(max_length=64, blank=True, default="")
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = "mini_program_user"

    def __str__(self):
        return self.client_id
