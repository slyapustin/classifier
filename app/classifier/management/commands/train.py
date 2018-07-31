from django.core.management.base import BaseCommand, CommandError
from classifier.train import train


class Command(BaseCommand):
    help = 'Train model'

    def handle(self, *args, **options):
        self.stdout.write('Start training...')
        train()
        self.stdout.write('Finished.')
