from django.contrib import admin
from django.urls import path, include

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', include('waveform.urls')),  # 引入 waveform 應用的路徑
]
