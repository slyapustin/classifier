import json
import random
import sys
import unicodedata

import nltk
import numpy as np
import tflearn
from nltk.stem.lancaster import LancasterStemmer

from config import MODEL_PATH, TENSORBOARD_PATH
from utils import init_network

# a table structure to hold the different punctuation used
tbl = dict.fromkeys(i for i in range(sys.maxunicode)
                    if unicodedata.category(chr(i)).startswith('P'))

# initialize the stemmer
stemmer = LancasterStemmer()

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
        # remove any punctuation from the sentence
        punctuation_free_sentence = sentence.translate(tbl)
        print(punctuation_free_sentence)
        # extract words from each sentence and append to the word list
        sentence_words = nltk.word_tokenize(punctuation_free_sentence)

        print("tokenized words: ", sentence_words)
        words.extend(sentence_words)
        docs.append((sentence_words, category))

# stem and lower each word and remove duplicates
words = [stemmer.stem(w.lower()) for w in words]
words = sorted(list(set(words)))

print(words)
print(docs)

# create our training data
training = []
# create an empty array for our output
output_empty = [0] * len(categories)


for doc in docs:
    # initialize our bag of words(bow) for each document in the list
    bow = []
    # list of tokenized words for the pattern
    token_words = doc[0]
    # stem each word
    token_words = [stemmer.stem(word.lower()) for word in token_words]
    # create our bag of words array
    for w in words:
        bow.append(1) if w in token_words else bow.append(0)

    output_row = list(output_empty)
    output_row[categories.index(doc[1])] = 1

    # our training set will contain a the bag of words model and the output row that tells
    # which catefory that bow belongs to.
    training.append([bow, output_row])

# shuffle our features and turn into np.array as tensorflow  takes in numpy array
random.shuffle(training)
training = np.array(training)

# trainX contains the Bag of words and train_y contains the label/ category
train_x = list(training[:, 0])
train_y = list(training[:, 1])

network = init_network(train_x, train_y)

# Define model and setup TensorBoard
model = tflearn.DNN(network, tensorboard_dir=TENSORBOARD_PATH)

# Start training (apply gradient descent algorithm)
model.fit(train_x, train_y, n_epoch=1000, batch_size=8, show_metric=True)
model.save(MODEL_PATH)


# a method that takes in a sentence and list of all words
# and returns the data in a form the can be fed to tensorflow


def get_tf_record(sentence):
    global words
    # tokenize the pattern
    sentence_words = nltk.word_tokenize(sentence)
    # stem each word
    sentence_words = [stemmer.stem(word.lower()) for word in sentence_words]
    # bag of words
    bow = [0]*len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bow[i] = 1

    return(np.array(bow))


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
