import tflearn

import numpy as np
from config import MODEL_PATH
from utils import get_categories, init_network, get_words, get_tf_record

categories = get_categories()
words = get_words()


# Define model and setup TensorBoard
network = init_network(len(words), len(categories))
model = tflearn.DNN(network)
model.load(MODEL_PATH)

while True:
    sentence = input("Enter sentence to test: ")
    tf_record = get_tf_record(words, sentence)
    category = categories[np.argmax(model.predict([tf_record]))]
    print("Category is: %s." % category)
