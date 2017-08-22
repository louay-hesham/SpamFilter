import os
import nltk
from nltk.corpus import stopwords
import json
from random import shuffle

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
        if os.path.split(directories)[1] == 'ham' or os.path.split(directories)[1] == 'spam':
            print("reading", directories, subdirs, len(files))
            for filename in files:
                if os.path.split(directories)[1] == 'ham':
                    ham_count += 1
                else:
                    spam_count += 1      
                with open(os.path.join(directories, filename), encoding="latin-1") as f:
                    text = f.read().lower()
                    words = tokenize_text(text)
                    if os.path.split(directories)[1] == 'ham':
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
        model[word] = { "ham": (count["ham"] + 1) / (ham_words_count + len(word_count)),
                            "spam": (count["spam"] + 1) / (spam_words_count + len(word_count))}
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

def get_unique_words(words):
    word_set = set()
    stop_words = set(stopwords.words('english'))
    for word in words:
        if word not in stop_words and word != "subject" and word != "":
            word_set.update([word])
    return word_set

def get_class_probability(model, words, p_class, type, count):
    p = p_class
    for word in words:
        if word in model:
            p *= model[word][type]
        else:
            p *= (1 / (count[type] + len(model)))
    return p    

def classify_email(model, p_ham, p_spam, ham_words_count, spam_words_count, file_path):
    count = {'ham': ham_words_count, 'spam': spam_words_count}
    with open(file_path, encoding="latin-1") as f:
        text = f.read().lower()
        return get_text_type(model, p_ham, p_spam, count, text)

def classify_batch_emails(model, p_ham, p_spam, ham_words_count, spam_words_count, file_path):
    count = {'ham': ham_words_count, 'spam': spam_words_count}
    with open(file_path, encoding="latin-1") as f:
        data = f.read().lower()
        emails = data.split('\n')
        return [get_text_type(model, p_ham, p_spam, count, email) for email in emails]

def classify_text(model, p_ham, p_spam, ham_words_count, spam_words_count, text):
    count = {'ham': ham_words_count, 'spam': spam_words_count}
    words = tokenize_text(text)
    unique_words = get_unique_words(words)
    ham = get_class_probability(model, unique_words, p_ham, "ham", count)
    spam = get_class_probability(model, unique_words, p_spam, "spam", count)
    return "ham" if ham > spam else "spam"


def calc_acc():
    (modeltest, p_ham_test, p_spam_test, ham_words_count_test, spam_words_count_test, test_list) = build_test_model()
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    for mail in test_list:
        if mail[1] == classify_text(modeltest, p_ham_test, p_spam_test, ham_words_count_test, spam_words_count_test, mail[0]):
            if mail[1] == "ham":
                tp +=1
            else:
                tn +=1
        else:
            if mail[1] == "ham":
                fn +=1
            else:
                fp +=1
    return (tp + tn) / (tp + tn + fp + fn)

def read_files():
    mail_list = []
    data_dir = "Data"
    #opening emails in training data
    #reading files into a list of tuples
    for directories, subdirs, files in os.walk(data_dir):
        if os.path.split(directories)[1] == 'ham' or os.path.split(directories)[1] == 'spam':
            print("reading", directories, subdirs, len(files))
            for filename in files:
                with open(os.path.join(directories, filename), encoding="latin-1") as f:
                    text = f.read().lower()
                    if os.path.split(directories)[1] == 'ham':
                        mail_list.append((text, "ham"))
                    else:
                        mail_list.append((text, "spam"))
    return mail_list

def train_data(mail_list):
    ham_words = []
    spam_words = []
    p_spam = 0
    p_ham = 0
    spam_count = 0
    ham_count = 0
    print("shuffling the mail list")
    shuffle(mail_list)
    train = int(len(mail_list) * .8)
    train_list = mail_list[:train]
    test_list = mail_list[train:]
    print("training data")
    for mail in train_list:
        if mail[1] == "ham":
            ham_count +=1
            ham_words += tokenize_text(mail[0])
        else:
            spam_count +=1
            spam_words += tokenize_text(mail[0])

    print("tokenized")
    total_count = spam_count + ham_count
    p_spam = spam_count / total_count
    p_ham = ham_count / total_count
    return (ham_words, spam_words, p_ham, p_spam, test_list)

def prepare_data():
    print("Reading training data")
    mail_list = read_files() 
    (ham_words, spam_words, p_ham, p_spam, test_list) = train_data(mail_list)
    print("spam probability is ", p_spam)
    print("ham probability is ", p_ham)
    print("creating dictionary")
    (word_count, ham_words_count, spam_words_count) = create_word_dict(ham_words, spam_words)
    print("Dictionary length is ", len(word_count))
    return (word_count, ham_words_count, spam_words_count, p_ham, p_spam, test_list)

def build_test_model():
    (word_count, ham_words_count, spam_words_count, p_ham, p_spam,test_list) = prepare_data()
    model = make_model(word_count, ham_words_count, spam_words_count)
    return (model, p_ham, p_spam, ham_words_count, spam_words_count, test_list)