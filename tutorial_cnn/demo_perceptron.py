import os
import numpy as np
from pandas import DataFrame
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re
import tensorflow as tf
from collections import Counter
import matplotlib.pyplot as plt

PUNCTUATION = nltk.download('punkt')
STOPWORDS = nltk.download('stopwords')


def preprocess(sentence):
    sentence = sentence.lower()
    clean_sentence = re.sub(r'[^\w\s]', '', sentence)
    tokens = word_tokenize(clean_sentence)
    filtered_words = [word for word in tokens if word not in set(stopwords.words('english'))]
    return filtered_words


def _load_doc(filename):
    with open(filename, 'r') as file:
        text = file.read()
        return text


def create_vocabulary(directory, vocab, flag):
    lines = list()
    for filename in os.listdir(directory):
        if flag and filename.startswith('cv9'):  # Skipping test set
            continue
        if not flag and not filename.startswith('cv9'):  # Skipping a review in training set
            continue
        clean_tokens = preprocess(_load_doc(os.path.join(directory, filename)))  # Load review & clean it
        if flag:
            vocab.update(clean_tokens)
        clean_tokens = [word for word in clean_tokens if word in vocab]
        line = ' '.join(clean_tokens)
        lines.append(line)
    return lines


def load_vocab_labels(vocab, flag):
    # Add all the files of the training dataset to the vocab
    neg = create_vocabulary('Polarity_Dataset/review_polarity/txt_sentoken/neg', vocab, flag)
    pos = create_vocabulary('Polarity_Dataset/review_polarity/txt_sentoken/pos', vocab, flag)
    docs = neg + pos
    # Create output labels, 0 for negative and 1 for positive
    labels = [0 for _ in range(len(neg))] + [1 for _ in range(len(pos))]
    return docs, labels


def create_tokenizer(lines):
    tokenizer = tf.keras.preprocessing.text.Tokenizer()
    tokenizer.fit_on_texts(lines)
    return tokenizer


def prepare_data(tokenizer, train_docs, test_docs, mode):
    # Encoding the training dataset
    X_train = tokenizer.texts_to_matrix(train_docs, mode=mode)
    # Encoding the testing dataset
    X_test = tokenizer.texts_to_matrix(test_docs, mode=mode)
    return X_train, X_test


def perceptron_model(n_words):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(50, input_shape=(n_words,), activation='relu'))
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
    # Compile model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model


def evaluate_model(X_train, y_train, X_test, y_test):
    scores = list()
    n_words = X_train.shape[1]
    n_repeats = 10  # 10-fold cross validation
    for i in range(n_repeats):
        # define the model
        model = perceptron_model(X_train.shape[1])
        # fit the model
        model.fit(X_train, np.array(y_train), epochs=10, verbose=0)
        # evaluate the model
        loss, accuracy = model.evaluate(X_test, np.array(y_test), verbose=0)
        print(f'Accuracy: {accuracy * 100.0:.3f}')
        scores.append(accuracy)
    return scores


def predict_sentiment(review, vocab, tokenizer, model):
    # 1. Clean the review
    tokens = preprocess(review)
    # 2. Filter words based on vocab
    tokens = [word for word in tokens if word in vocab]
    # 3. Encode the review
    encoded = tokenizer.texts_to_matrix([review], mode='binary')
    # 4. Predict the sentiment
    yhat = model.predict(encoded, verbose=0)
    # 5. Convert the score into appropriate label
    percent_pos = yhat[0, 0]
    if round(percent_pos) == 0:
        return (1 - percent_pos) * 100, 'Negative'
    return percent_pos*100, 'Positive'


def main():
    test_sentence = 'Hello, how are you? How is the weather today?'
    # Step 1: Convert sentence to lower case, remove punctuations and stopwords
    filtered_words = preprocess(test_sentence)
    print(filtered_words)

    # Step 2: Load the documents to create a vocabulary and corresponding labels
    vocab = Counter()
    train_docs, y_train = load_vocab_labels(vocab, True)
    test_docs, y_test = load_vocab_labels(vocab, False)

    print(f'Loaded documents (training): {len(train_docs)}')
    print(f'Loaded documents (testing): {len(test_docs)}')
    print(f'Vocabulary size: {len(vocab)}')

    # Convert the vocab into a set
    vocab = set(vocab)

    # Step 3: Convert our reviews into a Bag-of-Words vector using tokenizer
    tokenizer = create_tokenizer(train_docs)
    X_train, X_test = prepare_data(tokenizer, train_docs, test_docs, 'binary')
    print(X_train.shape)  # (1800, 45055)
    print(X_test.shape)

    # Step 4: Defining a model
    model = perceptron_model(X_train.shape[1])
    model.fit(X_train, np.array(y_train), epochs=10, verbose=0)
    model.summary()
    loss, accuracy = model.evaluate(X_test, np.array(y_test), verbose=0)
    print(f'Accuracy: {accuracy * 100}')

    # Step 5: Evaluate the accuracy of the model for all modes by performing 10-fold cross validation
    # binary = words marked as present(1) and not present(0)
    # count = occurrence count for each word is marked as an integer
    # tfidf = where words that appear across all documents are penalized
    # freq = words are scored based on their frequency within the document
    modes = ['binary', 'count', 'tfidf', 'freq']
    results = DataFrame()
    for mode in modes:
        X_train, X_test = prepare_data(tokenizer, train_docs, test_docs, mode)
        print(X_train.shape)
        print(X_test.shape)
        scores = evaluate_model(X_train, np.array(y_train), X_test, np.array(y_test))
        results[mode] = scores

    # Summarize the results
    print(results.describe())
    # Plot the results to compare the accuracy across all modes
    results.boxplot()
    plt.show()

    # Step 6: Predict Sentiment for Movie Reviews
    test1 = 'This is a bad movie'
    percent_1, sentiment_1 = predict_sentiment(test1, vocab, tokenizer, model)
    print(f'Review: {test1}')
    print(f'Sentiment: {sentiment_1} ({percent_1:.3f})')

    test2 = 'This is a great movie! It was so much fun watching it!!'
    percent_2, sentiment_2 = predict_sentiment(test2, vocab, tokenizer, model)
    print(f'Review: {test2}')
    print(f'Sentiment: {sentiment_2} ({percent_2:.3f})')


if __name__ == '__main__':
    main()
