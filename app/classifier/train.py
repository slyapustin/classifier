import json
import random

import numpy as np
import tflearn
from django.conf import settings

from classifier.utils import init_network, get_tokenized_words, get_categories, get_words


def train():
    # read the json file and load the training data
    with open(settings.CLASSIFIER_DATA_SET) as json_data:
        data = json.load(json_data)

    # get a list of all categories to train for
    categories = get_categories()
    words = get_words()

    # a list of tuples with words in the sentence and category name
    docs = []

    for category in categories:
        for sentence in data[category]:
            sentence_words = get_tokenized_words(sentence)
            print("tokenized words: ", sentence_words)
            docs.append((sentence_words, category))

    # create our training data
    training = []
    # create an empty array for our output
    output_empty = [0] * len(categories)

    for doc in docs:
        # initialize our bag of words for each document in the list
        # list of tokenized words for the pattern
        token_words = doc[0]

        # Create bag of words array
        bag_of_words = [1 if word in token_words else 0 for word in words]

        output_row = list(output_empty)
        output_row[categories.index(doc[1])] = 1

        # Our training set will contain a the bag of words model
        # and the output row that tells which category that bag of words belongs to.
        training.append([bag_of_words, output_row])

    # Shuffle our features and turn into np.array as TensorFlow  takes in numpy array
    random.shuffle(training)
    training = np.array(training)

    # x_inputs contains the Bag of words and y_targets contains the category
    x_inputs = list(training[:, 0])
    y_targets = list(training[:, 1])

    # Define model and setup TensorBoard
    x_size = len(x_inputs[0])
    y_size = len(y_targets[0])
    network = init_network(x_size, y_size)
    model = tflearn.DNN(network, tensorboard_dir=settings.CLASSIFIER_TENSORBOARD_PATH)
    # Start training (apply gradient descent algorithm)
    model.fit(x_inputs, y_targets, n_epoch=1000, batch_size=8, show_metric=True)
    model.save(settings.CLASSIFIER_MODEL_PATH)