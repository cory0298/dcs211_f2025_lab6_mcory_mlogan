from textblob import *
from textblob.classifiers import NaiveBayesClassifier
import nltk

train = [
    ("I love this sandwich. I also liked the smoothie.", "pos"),
    ("this is an amazing place!", "pos"),
    ("I feel very good about these beers.", "pos"),
    ("this is my favorite cafe.", "pos"),
    ("what an awesome view", "pos"),
    ("I do not like this restaurant", "neg"),
    ("I am tired of this stuff.", "neg"),
    ("I can't deal with the service", "neg"),
    ("the food was awful.", "neg"),
    ("my coffee was horrible.", "neg"),
    ("It was okay", "neu"),
    ("it's fine", "neu"),
    ("My salad had lettuce and tomatoes", "neu"),
    ("the restaurant is on lisbon street", "neu")
]
test = [
    ("the beer was good.", "pos"),
    ("I did not enjoy my salad", "neg"),
    ("I ain't feeling great today.", "neg"),
    ("I feel amazing!", "pos"),
    ("Gary is a friend of mine.", "pos"),
    ("I can't believe I'm doing this.", "neg"),
]
'''
for review in train:
    sentence = review[0]
    sentence = nltk.word_tokenize(sentence)
    sentence_pos = nltk.pos_tag(sentence)
    print(sentence_pos)
'''
cl = NaiveBayesClassifier(train)

cl.show_informative_features()
'''
for review in train:
    print(cl.classify(review[0]))
    prob_dist = cl.prob_classify(review[0])
    print(prob_dist)
    print(prob_dist.max())


for review in test:
    print(cl.classify(review[0]))
    prob_dist = cl.prob_classify(review[0])
    print(prob_dist)
    print(prob_dist.max())
'''