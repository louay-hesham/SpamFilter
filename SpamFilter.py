import os
import nltk
from nltk.corpus import stopwords
import json
from random import shuffle

def tokenize_text(data):
    words = nltk.tokenize.word_tokenize(data)
    stop_words = set(stopwords.words('english'))
    ps = nltk.stem.SnowballStemmer('english')
    filtered_list = ["" if x in stop_words else ps.stem(x) if x.isalpha() else "num" if x.isnumeric() else "" for x in words]
    return filtered_list
   
def read_training_data():
    data_dir = "TrainingData"
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
    
    for word in ham_words:
        if word != "subject" and word != "":
            ham_words_count += 1
            if word in word_count:
                word_count[word]["ham"] += 1
            else:
                word_count[word] = {"ham": 1, "spam": 0}

    for word in spam_words:
        if word != "subject" and word != "":
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
        return classify_text(model, p_ham, p_spam, ham_words_count, spam_words_count, text)

def classify_batch_emails(model, p_ham, p_spam, ham_words_count, spam_words_count, file_path):
    count = {'ham': ham_words_count, 'spam': spam_words_count}
    with open(file_path, encoding="latin-1") as f:
        data = f.read().lower()
        emails = data.split('\n')
        return [classify_text(model, p_ham, p_spam, ham_words_count, spam_words_count, email) for email in emails]

def classify_text(model, p_ham, p_spam, ham_words_count, spam_words_count, text):
    count = {'ham': ham_words_count, 'spam': spam_words_count}
    words = tokenize_text(text)
    unique_words = set(words)
    unique_words -= set(["subject", ""])
    ham = get_class_probability(model, unique_words, p_ham, "ham", count)
    spam = get_class_probability(model, unique_words, p_spam, "spam", count)
    return "ham" if ham > spam else "spam"

def read_test_files():
    mail_list = []
    data_dir = "TestData"
    for directories, subdirs, files in os.walk(data_dir):
        if os.path.split(directories)[1] == 'ham' or os.path.split(directories)[1] == 'spam':
            print("reading", directories, subdirs, len(files))
            for filename in files:
                with open(os.path.join(directories, filename), encoding="latin-1") as f:
                    text = f.read().lower()
                    mail_list.append((text, os.path.split(directories)[1]))
    return mail_list

def test_accuracy(model, p_ham, p_spam, ham_words_count, spam_words_count):
    print("Reading test data")
    mail_list = read_test_files()
    shuffle(mail_list)
    print("Test data size is", len(mail_list), "emails")
    n_samples = 5
    sample_size = int(len(mail_list) / n_samples)
    print("Number of samples =", n_samples)
    print("Sample size =", sample_size, "emails")
    avg_acc = 0
    for i in range(0, n_samples):
        test_list = mail_list[i * sample_size : (i + 1) * sample_size]
        tp = 0
        tn = 0
        fp = 0
        fn = 0
        for mail in test_list:
            if mail[1] == classify_text(model, p_ham, p_spam, ham_words_count, spam_words_count, mail[0]):
                if mail[1] == "ham":
                    tp +=1
                else:
                    tn +=1
            else:
                if mail[1] == "ham":
                    fn +=1
                else:
                    fp +=1
        acc = (tp + tn) / (tp + tn + fp + fn)
        print("True positive =", tp)
        print("True negative =", tn)
        print("False positive =", fp)
        print("False negative =", fn)
        print("Sample", i + 1, "accuracy =", acc)
        avg_acc += acc
    return (avg_acc / n_samples)