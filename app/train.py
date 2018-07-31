import json
import random

import numpy as np
import tflearn

from config import MODEL_PATH, TENSORBOARD_PATH
from utils import init_network, get_tokenize_words

# read the json file and load the training data
with open('sample_data.json') as json_data:
    data = json.load(json_data)

# get a list of all categories to train for
categories = list(data.keys())
words = []

# a list of tuples with words in the sentence and category name
docs = []

for category in data.keys():
    for sentence in data[category]:
        sentence_words = get_tokenize_words(sentence)
        print("tokenized words: ", sentence_words)
        words.extend(sentence_words)
        docs.append((sentence_words, category))

words = sorted(list(set(words)))

print(words)
print(docs)

# create our training data
training = []
# create an empty array for our output
output_empty = [0] * len(categories)


for doc in docs:
    # initialize our bag of words(bow) for each document in the list
    bag_of_words = []
    # list of tokenized words for the pattern
    token_words = doc[0]

    # create our bag of words array
    for w in words:
        bag_of_words.append(1) if w in token_words else bag_of_words.append(0)

    output_row = list(output_empty)
    output_row[categories.index(doc[1])] = 1

    # Our training set will contain a the bag of words model
    # and the output row that tells which category that bag of words belongs to.
    training.append([bag_of_words, output_row])

# Shuffle our features and turn into np.array as TensorFlow  takes in numpy array
random.shuffle(training)
training = np.array(training)

# trainX contains the Bag of words and train_y contains the label/ category
x_inputs = list(training[:, 0])
y_targets = list(training[:, 1])

network = init_network(x_inputs, y_targets)

# Define model and setup TensorBoard
model = tflearn.DNN(network, tensorboard_dir=TENSORBOARD_PATH)

# Start training (apply gradient descent algorithm)
model.fit(x_inputs, y_targets, n_epoch=1000, batch_size=8, show_metric=True)
model.save(MODEL_PATH)


# a method that takes in a sentence and list of all words
# and returns the data in a form the can be fed to tensorflow


def get_tf_record(sentence):
    global words
    # Tokenize the pattern and stem each word
    sentence_words = get_tokenize_words(sentence)

    bag_of_words = [0]*len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag_of_words[i] = 1

    return np.array(bag_of_words)


# Let's test our model
sentences = [
    "what time is it?",
    "I gotta go now",
    "do you know the time now?",
    "you must be a couple of years older then her!",
    "Hi there!",
]

for sentence in sentences:
    # we can start to predict the results for each of the 4 sentences
    print(sentence, ' = ', categories[np.argmax(model.predict([get_tf_record(sentence)]))])
