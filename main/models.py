from django.db import models

# Create your models here.
class postfile(models.Model):
    file_name = models.CharField(max_length=250)
    uploaded_date = models.DateTimeField()
    file_url = models.CharField(max_length=250)
    file_extension = models.CharField(max_length=6, null=True, blank=True)
    file_type = models.CharField(max_length = 6, blank=True, null=True) # this should be file type (audio, video, image, unkow) unknown
