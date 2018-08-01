import numpy as np
import tflearn
from django.conf import settings

from classifier.utils import get_categories, init_network, get_words, get_tf_record

categories = get_categories()
words = get_words()


def predict_category(text):
    # Define model and setup TensorBoard
    network = init_network(len(words), len(categories))
    model = tflearn.DNN(network)
    model.load(settings.CLASSIFIER_MODEL_PATH)

    tf_record = get_tf_record(words, text)
    return categories[np.argmax(model.predict([tf_record]))]
