import os
import nltk
from random import shuffle
from nltk.classify.naivebayes import NaiveBayesClassifier

def create_word_features(words):
    my_dict = dict( [ (word, True) for word in words] )
    return my_dict

data_dir = "Data"

ham_list = []
spam_list = []
 
for directories, subdirs, files in os.walk(data_dir):
    if (os.path.split(directories)[1]  == 'ham'):
        print(directories, subdirs, len(files))
        for filename in files:      
            with open(os.path.join(directories, filename), encoding="latin-1") as f:
                data = f.read()
                words = nltk.tokenize.word_tokenize(data)                
                ham_list.append((create_word_features(words), "ham"))
    
    if (os.path.split(directories)[1]  == 'spam'):
        print(directories, subdirs, len(files))
        for filename in files:
            with open(os.path.join(directories, filename), encoding="latin-1") as f:
                data = f.read()
                words = nltk.tokenize.word_tokenize(data)                
                spam_list.append((create_word_features(words), "spam"))

print(ham_list[0])
print(spam_list[0])

combined_list = ham_list + spam_list
 
shuffle(combined_list)
training_part = int(len(combined_list) * .7)
print(len(combined_list))
 
training_set = combined_list[:training_part]
 
test_set =  combined_list[training_part:]
 
print (len(training_set))
print (len(test_set))

# Create the Naive Bayes filter
 
classifier = NaiveBayesClassifier.train(training_set)
 
# Find the accuracy, using the test data
 
accuracy = nltk.classify.util.accuracy(classifier, test_set)
 
print("Accuracy is: ", accuracy * 100)