import sys
import unicodedata

import nltk
import tensorflow as tf
import tflearn
from nltk.stem.lancaster import LancasterStemmer

# initialize the stemmer
stemmer = LancasterStemmer()

# a table structure to hold the different punctuation used
tbl = dict.fromkeys(i for i in range(sys.maxunicode)
                    if unicodedata.category(chr(i)).startswith('P'))


def init_network(X_inputs, Y_targets):
    # reset underlying graph data
    tf.reset_default_graph()
    # Build neural network
    net = tflearn.input_data(shape=[None, len(X_inputs[0])])
    net = tflearn.fully_connected(net, 8)
    net = tflearn.fully_connected(net, 8)
    net = tflearn.fully_connected(net, len(Y_targets[0]), activation='softmax')
    net = tflearn.regression(net)

    return net


def get_tokenized_words(text):
    """
    Remove any punctuation from the text and return list of lowercase words
    """

    return [stemmer.stem(word.lower()) for word in nltk.word_tokenize(text.translate(tbl))]
