from django.conf import settings
from django.conf.urls.static import static
from django.urls import path
from main import views
app_name = "main"
urlpatterns = [
    path("", views.mainpage, name="mainpage"),
    path("result", views.mainpage, name="result"),
    #path("filehandler", views.filehandler, name="filehandler"),
]+ static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
