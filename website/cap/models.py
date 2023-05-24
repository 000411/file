from django.db import models
from PIL import Image

# Create your models here.
# from django.db import models

class ImageModel(models.Model):
    image = models.ImageField(upload_to='images/', blank=True)

    def __str__(self):
        return self.image.name
