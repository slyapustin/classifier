# -*- coding:utf-8 -*-
from django.contrib import admin

from classifier.models import Category, Sentence, Train


class CategoryAdmin(admin.ModelAdmin):
    list_display = ('title', 'created')
    search_fields = ('title', )


class SentenceAdmin(admin.ModelAdmin):
    list_display = ('category', 'text', 'created')
    search_fields = ('text', )
    list_filter = ('created', 'category')


class TrainAdmin(admin.ModelAdmin):
    list_display = ('duration', 'created',)


admin.site.register(Train, TrainAdmin)
admin.site.register(Category, CategoryAdmin)
admin.site.register(Sentence, SentenceAdmin)
