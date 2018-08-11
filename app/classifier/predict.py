import numpy as np
import tflearn
from django.conf import settings

from classifier.models import Train
from classifier.utils import init_network, get_tf_record


def predict_category(text):
    train = Train.objects.filter(finished__isnull=False).order_by('finished').first()
    if not train:
        return dict(
            success=False,
            message='You need to train your dragon first'
        )

    categories = train.categories
    words = train.words

    # Define model and setup TensorBoard
    network = init_network(len(words), len(categories))
    model = tflearn.DNN(network)
    model.load(settings.CLASSIFIER_MODEL_PATH)

    tf_record = get_tf_record(words, text)

    return dict(
        success=True,
        message=categories[np.argmax(model.predict([tf_record]))]
    )
