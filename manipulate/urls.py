from django.contrib import admin
from django.urls import path,include
from . import views
urlpatterns = [
    path("",views.check,name="checking"),
    path("from",views.form,name="checking"),
     path("train",views.train,name="checking"),
    path("predict", views.predict, name="checking")
]
