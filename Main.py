from SpamFilter import prepare_training_data, init_naive_bayes

(word_count, ham_words_count, spam_words_count) = prepare_training_data()
word_prob = init_naive_bayes(word_count, ham_words_count, spam_words_count)
