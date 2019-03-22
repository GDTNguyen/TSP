from django.conf.urls import url, include
from django.contrib import admin
from django.conf import settings
from django.conf.urls.static import static
from . import views

urlpatterns = [
	url(r'^$', views.home, name='home'),
	url(r'^landingpageTSP/$', views.landingpageTSP, name='lTSP'),

	url(r'^game/$', views.graphOutput, name='plotter'),
    url(r'^game/tour/$', views.getATour, name='tour'),
    url(r'^game/userTour/$', views.userTour, name='userTour'),
    url(r'^game/hiscore/', views.gethiscores, name='hiscore'),
    url(r'^game/subhiscore/', views.submitTour, name='subhiscore'),
    url(r'^game/googletour/', views.googCTSP, name='googletour'),

    url(r'^results/$', views.vsliresults, name='results'),

    url(r'^accounts/register/', views.register, name='register'),

]

if settings.DEBUG:
	urlpatterns += static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)