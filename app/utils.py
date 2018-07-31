import json
import sys
import unicodedata

import nltk
import numpy as np
import tensorflow as tf
import tflearn
from nltk.stem.lancaster import LancasterStemmer

# initialize the stemmer
stemmer = LancasterStemmer()

# a table structure to hold the different punctuation used
tbl = dict.fromkeys(i for i in range(sys.maxunicode)
                    if unicodedata.category(chr(i)).startswith('P'))


def init_network(x_size, y_size):
    # reset underlying graph data
    tf.reset_default_graph()
    # Build neural network
    net = tflearn.input_data(shape=[None, x_size])
    net = tflearn.fully_connected(net, 8)
    net = tflearn.fully_connected(net, 8)
    net = tflearn.fully_connected(net, y_size, activation='softmax')
    net = tflearn.regression(net)

    return net


def get_tokenized_words(text):
    """
    Remove any punctuation from the text and return list of lowercase words
    """

    return [stemmer.stem(word.lower()) for word in nltk.word_tokenize(text.translate(tbl))]


def get_categories():
    with open('sample_data.json') as json_data:
        data = json.load(json_data)

    return list(data.keys())


def get_words():
    # Get ordered list of unique words which was used to train model
    words_list = []
    with open('sample_data.json') as json_data:
        data = json.load(json_data)

    for category in data.keys():
        for sentence in data[category]:
            words_list.extend(get_tokenized_words(sentence))

    return sorted(list(set(words_list)))


def get_tf_record(words, sentence):
    # A method that takes in a sentence and list of all words
    # And returns the data in a form the can be fed to TensorFlow

    # Tokenize the pattern and stem each word
    sentence_words = get_tokenized_words(sentence)

    bag_of_words = [1 if word in sentence_words else 0 for word in words]

    return np.array(bag_of_words)
