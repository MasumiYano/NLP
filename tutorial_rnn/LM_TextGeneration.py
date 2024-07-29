import numpy as np
import tensorflow as tf
from os import access


def load_doc(filename):
    with open(filename, 'r') as file:
        file_text = file.read()
        return file_text


def vocab_encoder(raw_text, tokenizer):
    tokenizer.fit_on_texts([raw_text])
    return tokenizer.word_index, len(tokenizer.word_index) + 1


def generate_sequence_labels(raw_text, tokenizer):
    sequences = []
    labels = []
    encoded = tokenizer.texts_to_sequences([raw_text])[0]

    for i in range(50, len(encoded)):
        sequences.append(encoded[i - 50:i])
        labels.append(encoded[i])

    return sequences, labels


def my_model(vocab_size):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=50, input_length=50))
    model.add(tf.keras.layers.LSTM(units=100, return_sequences=True))
    model.add(tf.keras.layers.LSTM(units=100))
    model.add(tf.keras.layers.Dense(units=100, activation='relu'))
    model.add(tf.keras.layers.Dense(units=vocab_size, activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model, model.summary()


def train(model, X, y, epochs=500, verbose=2):
    model.fit(X, y, epochs=epochs, verbose=verbose)


def generate_text(model, seed_text, n_words, tokenizer):
    in_text, output_text = seed_text, seed_text

    for i in range(n_words):
        encoded = tokenizer.texts_to_sequences([in_text])[0]
        encoded = tf.keras.preprocessing.sequence.pad_sequences([encoded], maxlen=50, padding='pre')
        encoded = np.reshape(encoded, (1, 50))
        prediction = model.predict(encoded, verbose=0)
        next_index = np.argmax(prediction)
        for word, index in tokenizer.word_index.items():
            if next_index == index:
                output_text += ' ' + word
                in_text += ' ' + word
                in_text = ' '.join(in_text.split()[1:])
                break

    return output_text


def main():
    tokenizer = tf.keras.preprocessing.text.Tokenizer()
    raw_text = load_doc("Metamorphosis.txt")
    start_seq = "Impeccable"

    # Phase 1: Prepare the data set.
    vocab_index, vocab_size = vocab_encoder(raw_text, tokenizer)
    sequences, labels = generate_sequence_labels(raw_text, tokenizer)
    X, y = np.array(sequences), np.array(labels)
    X = tf.keras.preprocessing.sequence.pad_sequences(X, maxlen=50, padding='pre')
    y = tf.keras.utils.to_categorical(y, num_classes=vocab_size)

    # Phase 2: Define the model
    model, summary = my_model(vocab_size)
    tf.keras.utils.plot_model(model, to_file='model_mymodel.png', show_shapes=True)

    # Phase 3: Train the model
    train(model, X, y)

    # Phase 4: Generate the text
    generated_text = generate_text(model, start_seq, 500, tokenizer)
    print(f"Generate Text: {generated_text}")


if __name__ == '__main__':
    main()
