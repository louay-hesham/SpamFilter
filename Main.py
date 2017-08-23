from SpamFilter import build_model, classify_email, classify_batch_emails, test_accuracy, naive, maxent
import tkinter as tk
from tkinter import filedialog
import json
from random import shuffle


root = tk.Tk()  #Initing GUI window, only used to select a file
root.withdraw() #Closing the GUI window itself as it's not needed

print("Trying to load pre-generated model")
try:
    #Opens model from JSON file
    with open('model.JSON') as data_file:  
        model = json.load(data_file)
    with open("param.JSON") as data_file:
        d = json.load(data_file)
        (p_ham, p_spam, ham_words_count, spam_words_count) = (d['p_ham'], d['p_spam'], d['ham_words_count'], d['spam_words_count'])
    print("Model loaded from JSON file")
except (FileNotFoundError, TypeError, ValueError) as e:
    #JSON file can not be found or is unreadable
    #Model is generated from training data then saved to the JSON file
    print("Model can not be loaded from JSON file. Building model from scratch.")
    (model, p_ham, p_spam, ham_words_count, spam_words_count) = build_model()


while (True): #Main loop
    choice = input("""
Please make a choice
    1- Single file mode
    2- Batch mode
    3- Rebuild model
    4- Calculate accuracy
    5- built in naive
    6- svm
    0- Exit

    Your choice is: """)
    if choice == '0':
        break
    elif choice == '1':
        file_path = filedialog.askopenfilename()    #Requesting file to be processed
        type = classify_email(model, p_ham, p_spam, ham_words_count, spam_words_count, file_path)
        print("        Email is", type, "\n\n")
    elif choice == '2':
        file_path = filedialog.askopenfilename()    #Requesting file to be processed
        types = classify_batch_emails(model, p_ham, p_spam, ham_words_count, spam_words_count, file_path)
        for i in range(0, len(types)):
            print("        Email", i + 1, "is", types[i])
        print("\n\n")
    elif choice == '3':
        print("Rebuilding model")
        (model, p_ham, p_spam, ham_words_count, spam_words_count) = build_model()
    elif choice == '4':
        acc = test_accuracy(model, p_ham, p_spam, ham_words_count, spam_words_count)
        print("Average accuracy = ",acc)
    elif choice == '5':
        naive()
    elif choice == '6':
        maxent()
