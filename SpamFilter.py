import os
import nltk
from nltk.corpus import stopwords
from stemming.porter2 import stem
import json
from random import shuffle

#Method that splits a string into words. Also removes stop words, stems text and replaces any number with "num"
def tokenize_text(data):
    words = nltk.tokenize.word_tokenize(data)       #Extracting words
    stop_words = set(stopwords.words('english'))    #Stop words init
    filtered_list = ["" if x in stop_words else stem(x) if x.isalpha() else "num" if x.isnumeric() else "" for x in words]   #Filtering words
    return filtered_list
   
#Reads training data and returns some parameters needed to generate the model
def read_training_data():
    data_dir = "TrainingData"
    ham_words = []  #List containing all words in ham emails. Repeatition is allowed
    spam_words = [] #List containing all words in spam emails. Repeatition is allowed
    p_spam = 0      #P(spam)
    p_ham = 0       #P(ham)
    spam_count = 0  #Spam emails count
    ham_count = 0   #Ham emails count
    
    #opening emails in training data
    for directories, subdirs, files in os.walk(data_dir):
        #Only processing emails which are in either a ham folder or a spam folder
        if os.path.split(directories)[1] == 'ham' or os.path.split(directories)[1] == 'spam':
            print("reading", directories, subdirs, len(files))
            for filename in files:
                #Increasing ham/spam counter
                if os.path.split(directories)[1] == 'ham':
                    ham_count += 1
                else:
                    spam_count += 1      
                with open(os.path.join(directories, filename), encoding="latin-1") as f:
                    text = f.read().lower()     #Email text as a single string
                    words = tokenize_text(text) #Tokenizing email into words
                    #Adding tokenized words into its respective list
                    if os.path.split(directories)[1] == 'ham':
                        ham_words += words
                    else:
                        spam_words += words
    total_count = spam_count + ham_count
    p_spam = spam_count / total_count   #Calculating P(spam)
    p_ham = ham_count / total_count     #Calculating P(ham)
    return (ham_words, spam_words, p_ham, p_spam)

#Creates a single dictionary with all the words ever recorded. Each word is accompanied by two counters.
#One for how many times is repeated in ham emails, the other is the same but in spam emails.
def create_word_dict(ham_words, spam_words):
    word_count = {}         #Word dictionary
    ham_words_count = 0     #number of words recorded in ham emails
    spam_words_count = 0    #number of words recorded in spam emails
    
    for word in ham_words:
        if word != "subject" and word != "": #Eliminating the word "subject" and empty strings
            ham_words_count += 1
            #Updating dictionary
            if word in word_count:
                word_count[word]["ham"] += 1
            else:
                word_count[word] = {"ham": 1, "spam": 0}

    for word in spam_words:
        if word != "subject" and word != "": #Eliminating the word "subject" and empty strings
            spam_words_count += 1
            #Updating dictionary
            if word in word_count:
                word_count[word]["spam"] += 1
            else:
                word_count[word] = {"ham": 0, "spam": 1}
    return (word_count, ham_words_count, spam_words_count)

#Prepares dictionary for model generation
def prepare_training_data():
    print("Reading training data")
    (ham_words, spam_words, p_ham, p_spam) = read_training_data()
    print("spam probability is ", p_spam)
    print("ham probability is ", p_ham)
    print("creating dictionary")
    (word_count, ham_words_count, spam_words_count) = create_word_dict(ham_words, spam_words)
    print("Dictionary length is ", len(word_count))
    return (word_count, ham_words_count, spam_words_count, p_ham, p_spam)

#Generates Naive Bayes model
def make_model(word_count, ham_words_count, spam_words_count):
    model = { } #Dictionary same as word_count, but with probabilities instead of a counter. See create_word_dict for more info.
    for word, count in word_count.items():
        #Calculating probability of each word
        model[word] = { "ham": (count["ham"] + 1) / (ham_words_count + len(word_count)),
                            "spam": (count["spam"] + 1) / (spam_words_count + len(word_count))}
    return model

#Builds the model from scratch
def build_model():
    #Reads training data and prepares the dictionary for later use
    (word_count, ham_words_count, spam_words_count, p_ham, p_spam) = prepare_training_data()
    #Builds the Naive Bayes model
    model = make_model(word_count, ham_words_count, spam_words_count)
    #Saving the model and other parameters in JSON files for later use
    with open('model.JSON', 'w') as model_file:
        json.dump(model, model_file)
    with open('param.JSON', 'w') as param_file:
        d = {"p_ham": p_ham, "p_spam": p_spam, "ham_words_count": ham_words_count, "spam_words_count": spam_words_count}
        json.dump(d, param_file)
    return (model, p_ham, p_spam, ham_words_count, spam_words_count)

#Returns P(email | class) for the specified class (Spam/Ham) and email
def get_class_probability(model, words, p_class, type, count):
    p = p_class
    for word in words:
        if word in model:
            p *= model[word][type]
        else:
            p *= (1 / (count[type] + len(model)))
    return p    

#Returns the class of an email
def classify_email(model, p_ham, p_spam, ham_words_count, spam_words_count, file_path):
    count = {'ham': ham_words_count, 'spam': spam_words_count}
    with open(file_path, encoding="latin-1") as f:
        text = f.read().lower()
        return classify_text(model, p_ham, p_spam, ham_words_count, spam_words_count, text)

#Returns the classes of a batch of emails
def classify_batch_emails(model, p_ham, p_spam, ham_words_count, spam_words_count, file_path):
    count = {'ham': ham_words_count, 'spam': spam_words_count}
    with open(file_path, encoding="latin-1") as f:
        data = f.read().lower()
        emails = data.split('\n')
        return [classify_text(model, p_ham, p_spam, ham_words_count, spam_words_count, email) for email in emails]

#Returns the class of a string
def classify_text(model, p_ham, p_spam, ham_words_count, spam_words_count, text):
    count = {'ham': ham_words_count, 'spam': spam_words_count} #Count of words in both classes
    words = tokenize_text(text)     #Tokenizing the string
    unique_words = set(words)       #Extracting unique words
    unique_words -= set(["subject", ""]) #Removing "Subject" and empty strings from the email
    ham = get_class_probability(model, unique_words, p_ham, "ham", count)       #Calculating P(text | ham)
    spam = get_class_probability(model, unique_words, p_spam, "spam", count)    #Calculating P(text | spam)
    return "ham" if ham > spam else "spam" #returning a string with the class of higher probability (argmax)

#Reads all test data to calculate the accuracy of the Naive Bayes model. Returns a list of tuples of (email, label)
def read_test_files():
    mail_list = []
    data_dir = "TestData"
    for directories, subdirs, files in os.walk(data_dir): #Opening directory
        #only checking ham and spam folders
        if os.path.split(directories)[1] == 'ham' or os.path.split(directories)[1] == 'spam':
            print("reading", directories, subdirs, len(files))
            for filename in files:
                with open(os.path.join(directories, filename), encoding="latin-1") as f:
                    text = f.read().lower()
                    mail_list.append((text, os.path.split(directories)[1]))
    return mail_list

#Returns average accuracy of the Naive Bayes model
def test_accuracy(model, p_ham, p_spam, ham_words_count, spam_words_count):
    print("Reading test data")
    mail_list = read_test_files()
    shuffle(mail_list) #Randomly shuffling test data for testing
    print("Test data size is", len(mail_list), "emails")
    n_samples = 5
    sample_size = int(len(mail_list) / n_samples)
    print("Number of samples =", n_samples)
    print("Sample size =", sample_size, "emails")
    avg_acc = 0
    #Dividing the shuffled test data into n samples. The accuracy of every sample is calculated alone then the average accuracy is calculated
    for i in range(0, n_samples):
        test_list = mail_list[i * sample_size : (i + 1) * sample_size] #Extracting sample from the test data
        tp = 0 #True positive count
        tn = 0 #True negative count
        fp = 0 #False positive count
        fn = 0 #False negative count
        for mail in test_list:
            if mail[1] == classify_text(model, p_ham, p_spam, ham_words_count, spam_words_count, mail[0]):
                #Email classification is correct
                if mail[1] == "ham":
                    tp +=1
                else:
                    tn +=1
            else:
                #Email classification is wrong
                if mail[1] == "ham":
                    fn +=1
                else:
                    fp +=1

        acc = (tp + tn) / (tp + tn + fp + fn)   #Accuracy calculation
        print("True positive =", tp)
        print("True negative =", tn)
        print("False positive =", fp)
        print("False negative =", fn)
        print("Sample", i + 1, "accuracy =", acc)
        avg_acc += acc
    return (avg_acc / n_samples)