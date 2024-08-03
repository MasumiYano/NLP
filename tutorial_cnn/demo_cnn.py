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


def encode_dogs(tokenizer, max_length, docs):
    # integer encode
    encoded = tokenizer.texts_to_sequences(docs)
    # pad input
    padded = tf.keras.preprocessing.sequence.pad_sequences(encoded, maxlen=max_length, padding='post')
    return padded


def CNN_model(vocab_size, max_length):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Embedding(vocab_size, 100, input_length=max_length))
    model.add(tf.keras.layers.Conv1D(filters=32, kernel_size=8, activation='relu'))
    model.add(tf.keras.layers.MaxPool1D(pool_size=2))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(10, activation='relu'))
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

    # compile the model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.summary()
    tf.keras.utils.plot_model(model, to_file='model_CNN.png', show_shapes=True)
    return model


def predict_sentiment(review, vocab, tokenizer, max_size, model):
    # 1. Clean the review
    tokens = preprocess(review)
    # 2. Filter words based on vocab
    tokens = [word for word in tokens if word in vocab]
    line = ' '.join(tokens)
    # 3. Encode the review
    encoded = encode_dogs(tokenizer, max_size, [line])
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
    vocab = set(vocab)
    train_docs, y_train = load_vocab_labels(vocab, True)
    test_docs, y_test = load_vocab_labels(vocab, False)

    print(f'Loaded documents (training): {len(train_docs)}')
    print(f'Loaded documents (testing): {len(test_docs)}')
    print(f'Vocabulary size: {len(vocab)}')

    # Step 3: Convert our reviews into a vector using tokenizer and encoding
    tokenizer = create_tokenizer(train_docs)
    # Define the vocab size for embedding layer
    vocab_size = len(tokenizer.word_index) + 1
    # Calculate max length of the sequences (reviews)
    max_length = max([len(sentence.split()) for sentence in train_docs])
    print(f'Max length: {max_length}')
    # Encode the input sequences
    X_train = encode_dogs(tokenizer, max_length, train_docs)
    X_test = encode_dogs(tokenizer, max_length, test_docs)
    print(f'X train shape {X_train.shape}')
    print(f'X test shape {X_test.shape}')

    # Step 4: Define the NN for Sentiment Analysis
    # Model 2: CNN model
    model = CNN_model(vocab_size, max_length)
    # Fit the model
    history = model.fit(X_train, np.array(y_train), epochs=10, verbose=2, validation_split=0.2)
    loss_train, accuracy_train = model.evaluate(X_train, np.array(y_train), verbose=0)
    print(f'Accuracy during training: {accuracy_train * 100}')
    loss_test, accuracy_test = model.evaluate(X_test, np.array(y_test), verbose=0)
    print(f'Accuracy during testing: {accuracy_test * 100}')

    # Step 5: Implement Early Stopping to handle overfitting
    # early_stopping = allow you to specify the performance measure to monitor.
    # Once it's triggerd, it stops the training
    # monitor = allows us to specify the performance measure
    # mode = need to specify weather the objectives is to increase(validation accuracy) or to decrease(validation loos)
    # patience = number of epochs to wait before stopping
    # min_delta = any change in performance, no matter how small will consider an improved
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', min_delta=1, verbose=1,
                                                      patience=10, restore_best_weights=True)
    history = model.fit(X_train, np.array(y_train), epochs=10, verbose=0, validation_split=0.2,
                        callbacks=[early_stopping])

    # plot for loss
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='validation')
    plt.title('CNN Model Loss')
    plt.ylabel('loss')
    plt.xlabel('epochs')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()

    # Step 6: Predict Sentiment for Movie Reviews
    test1 = 'This is a bad movie'
    percent_1, sentiment_1 = predict_sentiment(test1, vocab, tokenizer, max_length, model)
    print(f'Review: {test1}')
    print(f'Sentiment: {sentiment_1} ({percent_1:.3f})')

    test2 = 'This is a great movie! It was so much fun watching it!!'
    percent_2, sentiment_2 = predict_sentiment(test2, vocab, tokenizer, max_length, model)
    print(f'Review: {test2}')
    print(f'Sentiment: {sentiment_2} ({percent_2:.3f})')


if __name__ == '__main__':
    main()
