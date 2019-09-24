# -*- coding:utf-8 -*-
import json

from django.db import models
from django.utils import timezone
from django.utils.functional import cached_property


class Category(models.Model):
    title = models.CharField(max_length=50, unique=True)
    processed = models.BooleanField(default=False)

    created = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.title


class Sentence(models.Model):
    category = models.ForeignKey(Category, on_delete=models.CASCADE)
    text = models.TextField()
    processed = models.BooleanField(default=False)

    created = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.text


class Train(models.Model):
    data = models.TextField(help_text='JSON data for categories and words', null=True, blank=True)

    started = models.DateTimeField(null=True, blank=True)
    finished = models.DateTimeField(null=True, blank=True)

    created = models.DateTimeField(auto_now_add=True)

    @cached_property
    def words(self):
        if self.data is None:
            return []

        return json.loads(self.data).get('words', [])

    @cached_property
    def categories(self):
        if self.data is None:
            return []
        return json.loads(self.data).get('categories', [])

    @property
    def duration(self):
        if not self.started:
            return None

        if self.finished:
            return self.finished - self.started
        else:
            return timezone.now() - self.started
