from SpamFilter import prepare_training_data, build_naive_bayes_model

(word_count, ham_words_count, spam_words_count) = prepare_training_data()
model = build_naive_bayes_model(word_count, ham_words_count, spam_words_count)
print(model)