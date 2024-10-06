from django.urls import path
from .views import waveform_view, update_stft_image

urlpatterns = [
    path('', waveform_view, name='waveform-view'),
    path('stft/', update_stft_image, name='update-stft'),
]
