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

def processText(text: str) -> str:
    '''
    helper function to break down text into important words basedd on parts of speech
    parameters:
        text: given sentence review.
    returns: 
        broken down sentence review. 
    '''
    tokens = nltk.word_tokenize(text)
    tagged = nltk.pos_tag(tokens)
    keep = [w for w, pos in tagged if pos not in ('PRP', 'DT', 'IN')]
    return " ".join(keep)


# This part creates the new training set that has only the important words. It gets rid of things like it/this/that to prevent the model from getting confused
new_train_set = []
for text, label in train:
    processed = processText(text)
    new_train_set.append((processed, label))


#trains the model on the new set
cl = NaiveBayesClassifier(new_train_set)

#cl.show_informative_features()

#training set
for text, _ in train:
    prediction = cl.classify(processText(text))
    print(f"Review: {text}\n → Predicted: {prediction}\n")

#testing set
for text, actual in test:
    prediction = cl.classify(processText(text))
    print(f"Review: {text}\nActual: {actual} | Predicted: {prediction}\n")

#accuracy checker
test_processed = [(processText(t), label) for t, label in test]
accuracy = cl.accuracy(test_processed)
print(f"Classifier accuracy on test set: {accuracy}")

#actual utility, asks you to provide reviews that are then classified, quits when you write quit
print("\nEnter a restaurant/meal review.")
print("Type 'quit' to exit.\n")

while True:
    user_text = input("Your review: ")
    if user_text.lower().strip() == "quit":
        break

    processed = processText(user_text)
    print("Prediction:", cl.classify(processed))
    print()

