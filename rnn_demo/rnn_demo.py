import numpy as np
import tensorflow as tf


def load_doc(filename):
    with open(filename, 'r') as file:
        file_text = file.read()
        return file_text


def main():
    raw_text = load_doc('rhyme.txt')
    # Step 1: Create vocabulary
    unique_chars = sorted(list(set(raw_text)))
    print(unique_chars)
    print(len(unique_chars))


if __name__ == '__main__':
    main()
