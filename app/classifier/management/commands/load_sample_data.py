import json

from django.conf import settings
from django.core.management.base import BaseCommand

from classifier.models import Category, Sentence


class Command(BaseCommand):
    help = 'Load Sample data in to the DB'

    def handle(self, *args, **options):
        self.stdout.write('Start loading...')
        with open(settings.CLASSIFIER_DATA_SET, 'r+') as f:
            data = json.loads(f.read())

        for category, sentences in data.items():
            category_object, created = Category.objects.get_or_create(title=category)
            for sentence in sentences:
                Sentence.objects.create(
                    category=category_object,
                    text=sentence
                )
        self.stdout.write('Loaded.')
