import os
import nltk
from nltk.corpus import stopwords


def read_training_data():
    data_dir = "Data"
    ham_emails = []
    spam_emails = []
    ham_words = []
    spam_words = []
    p_spam = 0
    p_ham = 0
    spam_count = 0
    ham_count = 0
    stop_words = set(stopwords.words('english'))
    #opening emails in training data
    for directories, subdirs, files in os.walk(data_dir):
        #reading ham emails
        if (os.path.split(directories)[1]  == 'ham'):
            ham_count += 1
            print("reading ", directories, subdirs, len(files))
            for filename in files:      
                with open(os.path.join(directories, filename), encoding="latin-1") as f:
                    data = f.read().lower()
                    words = nltk.tokenize.word_tokenize(data)              
                    ham_emails.append([i for i in words if i not in stop_words])
                    ham_words += words
    
        #reading spam emails
        if (os.path.split(directories)[1]  == 'spam'):
            spam_count += 1
            print("reading ", directories, subdirs, len(files))
            for filename in files:
                with open(os.path.join(directories, filename), encoding="latin-1") as f:
                    data = f.read().lower()
                    words = nltk.tokenize.word_tokenize(data.lower())                
                    spam_emails.append([i for i in words if i not in stop_words])
                    spam_words += words
    
    total_count = spam_count + ham_count
    p_spam = spam_count / total_count
    p_ham = ham_count / total_count
    return (ham_words, spam_words, p_ham, p_spam)

def create_word_dict(ham_words, spam_words):
    word_count = {}
    ham_words_count = 0
    spam_words_count = 0
    for word in ham_words:
        if word.isalpha() and len(word) > 1:
            ham_words_count += 1
            if word in word_count:
                word_count[word]["ham"] += 1
            else:
                word_count[word] = {"ham": 1, "spam": 0}

    for word in spam_words:
        if word.isalpha() and len(word) > 1:
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
    return (word_count, ham_words_count, spam_words_count)

def init_naive_bayes(word_count, ham_words_count, spam_words_count):
    word_prob = { }
    for word, count in word_count.items():
        word_prob[word] = { "ham": (count["ham"] + 1)/(ham_words_count + len(word_count)),
                            "spam": (count["spam"] + 1)/(spam_words_count + len(word_count))}
    return word_prob
    



