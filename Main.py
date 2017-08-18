import os
import nltk

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