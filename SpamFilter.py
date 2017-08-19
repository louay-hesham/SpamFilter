import os
import nltk

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
    #opening emails in training data
    for directories, subdirs, files in os.walk(data_dir):
        #reading ham emails
        if (os.path.split(directories)[1]  == 'ham'):
            ham_count += 1
            print("reading ", directories, subdirs, len(files))
            for filename in files:      
                with open(os.path.join(directories, filename), encoding="latin-1") as f:
                    data = f.read()
                    words = nltk.tokenize.word_tokenize(data)                
                    ham_emails.append(words)
                    ham_words += words
    
        #reading spam emails
        if (os.path.split(directories)[1]  == 'spam'):
            spam_count += 1
            print("reading ", directories, subdirs, len(files))
            for filename in files:
                with open(os.path.join(directories, filename), encoding="latin-1") as f:
                    data = f.read()
                    words = nltk.tokenize.word_tokenize(data)                
                    spam_emails.append(words)
                    ham_words += words
    
    total_count = spam_count + ham_count
    p_spam = spam_count / total_count
    p_ham = ham_count / total_count
    return (ham_words, spam_words, p_ham, p_spam)

def create_word_dict(ham_words, spam_words):
    dictionary = {}
    for word in ham_words:
        if word.isalpha() and len(word) > 1:
            if word in dictionary:
                dictionary[word] = (dictionary[word][0] + 1, 0)
            else:
                dictionary[word] = (1, 0)

    for word in spam_words:
        if word.isalpha() and len(word) > 1:
            if word in dictionary:
                dictionary[word] = (dictionary[word][0], dictionary[word][1] + 1)
            else:
                dictionary[word] = (0, 1)
    return dictionary

def prepare_training_data():
    print("Reading training data")
    (ham_words, spam_words, p_ham, p_spam) = read_training_data()
    print("spam probability is ", p_spam)
    print("ham probability is ", p_ham)
    print("creating dictionary")
    dictionary = create_word_dict(ham_words, spam_words)
    print("Dictionary length is ", len(dictionary))



