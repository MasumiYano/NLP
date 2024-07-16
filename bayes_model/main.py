import os
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from collections import defaultdict

PUNCTUATION = ['.', ',', '?', '!', ':', ';', '\'', '"', '•', '(', ')', '[', ']', '{', '}', '\\', '/']


def process_data(review):
    cleaned_review = []
    text = word_tokenize(review)
    for word in text:
        if word not in stopwords.words('english') and word not in PUNCTUATION:
            cleaned_review.append(word)

    return cleaned_review


def count_reviews(directory_path):
    return_dict = defaultdict(int)
    for item in os.listdir(directory_path):
        item_file = open(f"{directory_path}/{item}", 'rt')
        item_text = item_file.read()
        item_file.close()
        cleaned_data = process_data(item_text)
        for word in cleaned_data:
            if directory_path == 'pos':
                return_dict[(word, 1)] += 1
            else:
                return_dict[(word, 0)] += 1
    return return_dict


def main():
    cleaned_data = process_data("This is a great movie !")
    print(cleaned_data)
    happy_dict = count_reviews("pos")
    negative_dict = count_reviews("neg")

    neg_words = {key[0] for key in negative_dict.keys()}

    counter = 0

    for key in happy_dict.keys():
        word = key[0]
        if word in neg_words:
            print(f"{word} is in the both dictionary.")
            counter += 1

    print(f"There are {counter} words that are in the both dictionary.")


if __name__ == '__main__':
    main()
