import os
import nltk
from nltk.corpus import stopwords
import json


def tokenize_text(data):
    words = nltk.tokenize.word_tokenize(data)
    ps = nltk.stem.SnowballStemmer('english')
    filtered_list = [ps.stem(x) if x.isalpha() else "num" if x.isdecimal() else "" for x in words]
    return filtered_list
   
def read_training_data():
    data_dir = "Data"
    ham_words = []
    spam_words = []
    p_spam = 0
    p_ham = 0
    spam_count = 0
    ham_count = 0
    
    #opening emails in training data
    for directories, subdirs, files in os.walk(data_dir):
        if os.path.split(directories)[1]  == 'ham' or os.path.split(directories)[1]  == 'spam':
            print("reading", directories, subdirs, len(files))
            for filename in files:
                if os.path.split(directories)[1]  == 'ham':
                    ham_count += 1
                else:
                    spam_count += 1      
                with open(os.path.join(directories, filename), encoding="latin-1") as f:
                    words = tokenize_text(f.read().lower())
                    if os.path.split(directories)[1]  == 'ham':
                        ham_words += words
                    else:
                        spam_words += words
    total_count = spam_count + ham_count
    p_spam = spam_count / total_count
    p_ham = ham_count / total_count
    return (ham_words, spam_words, p_ham, p_spam)

def create_word_dict(ham_words, spam_words):
    word_count = {}
    ham_words_count = 0
    spam_words_count = 0
    stop_words = set(stopwords.words('english'))
    for word in ham_words:
        if word not in stop_words and word != "subject" and word != "":
            ham_words_count += 1
            if word in word_count:
                word_count[word]["ham"] += 1
            else:
                word_count[word] = {"ham": 1, "spam": 0}

    for word in spam_words:
        if word not in stop_words and word != "subject" and word != "":
            spam_words_count += 1
            if word in word_count:
                word_count[word]["spam"] += 1
            else:
                word_count[word] = {"ham": 0, "spam": 1}
    return (word_count, ham_words_count, spam_words_count)

def prepare_training_data():
    print("Reading training data")
    (ham_words, spam_words, p_ham, p_spam) = read_training_data()
    print("spam probability is ", p_spam)
    print("ham probability is ", p_ham)
    print("creating dictionary")
    (word_count, ham_words_count, spam_words_count) = create_word_dict(ham_words, spam_words)
    print("Dictionary length is ", len(word_count))
    return (word_count, ham_words_count, spam_words_count, p_ham, p_spam)

def make_model(word_count, ham_words_count, spam_words_count):
    model = { }
    for word, count in word_count.items():
        model[word] = { "ham": (count["ham"] + 1)/(ham_words_count + len(word_count)),
                            "spam": (count["spam"] + 1)/(spam_words_count + len(word_count))}
    return model

def build_model():
    (word_count, ham_words_count, spam_words_count, p_ham, p_spam) = prepare_training_data()
    model = make_model(word_count, ham_words_count, spam_words_count)
    with open('model.JSON', 'w') as model_file:
        json.dump(model, model_file)
    with open('param.JSON', 'w') as param_file:
        d = {"p_ham": p_ham, "p_spam": p_spam, "ham_words_count": ham_words_count, "spam_words_count": spam_words_count}
        json.dump(d, param_file)
    return (model, p_ham, p_spam, ham_words_count, spam_words_count)
    



