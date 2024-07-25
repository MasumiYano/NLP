# -*- coding: utf-8 -*-
"""WordLM1.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1uOnODEutWwUTR1Md0HOS9nyu7qZzCl9v
"""
'''
Author: Raghavi Sakpal
Date: June, 2024
Description: Word-based text generation (Language Model Problem) using RNN and LSTM.
             Input = one-word in, Output = one-word out
             Dataset = rhyme.txt
             This code is using the Keras Tokenizer for encoding the data & Embedding layer to conver input into vector representation.
Note: We need punctuations for text generation and no need to make words case-sensitive.
'''

import numpy as np
import tensorflow as tf


# Code to load the document in memory
def load_doc(filename):
    file = open(filename, 'r')
    text = file.read()
    file.close()
    return text


# Load the rhyme.txt file and print it to the screen
raw_text = load_doc('Metamorphosis.txt')

# Step 1: Create vocab and encode the words to integers
# We will use Keras Tokenizer to do so...
tokenizer = tf.keras.preprocessing.text.Tokenizer()
# fit_on_texts: creates the vocabulary index based on word frequency.
tokenizer.fit_on_texts([raw_text])
print(tokenizer.word_index)

# Let's store the vocab length
# Note: +1 is to make sure we are not out of bounds
vocab_size = len(tokenizer.word_index) + 1
print('Vocabulary Size: %d' % vocab_size)

# Step 2: Create input sequences and corresponding labels
# texts_to_sequences: transforms each text in texts to a sequence of integers. So it basically takes each word in the text and replaces it with its corresponding integer value from the word_index dictionary.
encoded = tokenizer.texts_to_sequences([raw_text])[0]
print(encoded)
sequences = []
labels = []

# Note: We are implementing one-word in and one-word out LM
for i in range(1, len(encoded)):
    sequences.append(encoded[i - 1])  # input: word 1
    labels.append(encoded[i])  # output: word 2
print('Total Sequences: %d' % len(sequences))
print(sequences)
print(labels)

# Step 3: Encode text to one-hot vector representation
# Convert the sequences & labels into numpy arrays
X = np.array(sequences)
Y = np.array(labels)

# Convert the labels into one-hot vector representation
# Note: Sequences will be convered to one-hot vector using embedding layer
Y = tf.keras.utils.to_categorical(Y, num_classes=vocab_size)

print(Y.shape)
print(Y)


# Step 4.a: Define a RNN model for Word-based text generation
# Model 1: Embedding Layer + Simple RNN with Relu and softmax functions
def model_1():
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Embedding(vocab_size, 10, input_length=1))  # Note: 10 is the output dimension
    model.add(tf.keras.layers.SimpleRNN(75, activation='relu'))
    model.add(tf.keras.layers.Dense(vocab_size, activation='softmax'))

    '''Compile the model for training using the following parameters:
  Optimization: Adam optimizer (used to minimize the loss function during the training process)
  Loss Function: Categorical Crossentropy
  Metrics: Accuracy'''
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    # summarize defined model
    model.summary()
    tf.keras.utils.plot_model(model, to_file='model_RNN1.png', show_shapes=True)
    return model


# Model 2: LSTM and softmax function
def model_2():
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Embedding(vocab_size, 10, input_length=1))
    model.add(tf.keras.layers.LSTM(75))
    model.add(tf.keras.layers.Dense(vocab_size, activation='softmax'))

    '''Compile the model for training using the following parameters:
  Optimization: Adam optimizer (used to minimize the loss function during the training process)
  Loss Function: Categorical Crossentropy
  Metrics: Accuracy'''
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    # summarize defined model
    model.summary()
    tf.keras.utils.plot_model(model, to_file='model_LSTM1.png', show_shapes=True)
    return model


# Step 5: Train the model using input sequences (X) and corresponding labels (Y) for 100 epochs (cycles)
model_RNN = model_1()
model_RNN.fit(X, Y, epochs=500, verbose=2)

model_LSTM = model_2()
model_LSTM.fit(X, Y, epochs=500, verbose=2)

# Step 6: Model Prediction
start_seq = 'Four'


# Function to generate the next set of words based on the initial seed text
def generate_text(model, tokenizer, seed_text, n_words):
    in_text, output_text = seed_text, seed_text

    for i in range(n_words):
        # encode the text into integers using Tokenizer
        encoded = np.array(tokenizer.texts_to_sequences([in_text])[0])
        # Use the trained model to make prediction
        prediction = model.predict(encoded, verbose=0)
        next_index = np.argmax(prediction)
        # map the predicted index back to word
        for word, index in tokenizer.word_index.items():
            if next_index == index:
                output_text += ' ' + word
                in_text = word
                break

    return output_text


# Generate text using both RNN & LSTM
generated_text_RNN = generate_text(model_RNN, tokenizer, start_seq, 10)
print("Generate Text using RNN: ", generated_text_RNN)

generated_text_LSTM = generate_text(model_LSTM, tokenizer, start_seq, 10)
print("Generate Text using RNN: ", generated_text_LSTM)
'''
Note: Since we are using argmax after few words the pattern repeats. Solution: many-words in or use other embedding techniques.
'''
