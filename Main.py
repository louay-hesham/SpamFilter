from SpamFilter import build_model, classify_email
import tkinter as tk
from tkinter import filedialog
import json


root = tk.Tk()
root.withdraw()

try:
    with open('model.JSON') as data_file:  
        model = json.load(data_file)
    with open("param.JSON") as data_file:
        d = json.load(data_file)
        (p_ham, p_spam, ham_words_count, spam_words_count) = (d['p_ham'], d['p_spam'], d['ham_words_count'], d['spam_words_count'])
    print("Model loaded from JSON file")
except (FileNotFoundError, TypeError, ValueError) as e:
    #JSON file can not be found or is unreadable
    print("Model can not be loaded from JSON file. Building model from scratch.")
    (model, p_ham, p_spam, ham_words_count, spam_words_count) = build_model()


while (True):
    choice = input("""Please make a choice
    1- Single file mode
    2- Batch mode
    3- Rebuild model
    0- Exit

    Your choice is: """)
    if choice == '0':
        break
    elif choice == '1':
        file_path = filedialog.askopenfilename()
        type = classify_email(model, p_ham, p_spam, ham_words_count, spam_words_count, file_path)
        print("Email is", type, "\n\n")
    elif choice == '3':
        (model, p_ham, p_spam, ham_words_count, spam_words_count) = build_model()
