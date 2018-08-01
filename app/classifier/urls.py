# -*- coding:utf-8 -*-
from django.conf.urls import url

from classifier import views

app_name = 'classifier'

urlpatterns = [
    url(r'^$', views.IndexView.as_view(), name='index'),
]
