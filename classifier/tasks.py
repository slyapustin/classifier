import logging

from celery import shared_task

logger = logging.getLogger(__name__)


@shared_task
def run_train():
    from classifier.train import train
    train()
