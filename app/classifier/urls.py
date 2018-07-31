# -*- coding:utf-8 -*-
from django.conf.urls import url

from classifier import views

app_name = 'classifier'

urlpatterns = [
    url(r'^$', views.IndexView.as_view(), name='index'),
    url(r'^train$', views.TrainView.as_view(), name='train'),
]
