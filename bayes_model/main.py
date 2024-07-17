import math
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
        with open(os.path.join(directory_path, item), 'rt') as item_file:
            item_text = item_file.read()
            cleaned_data = process_data(item_text)
            for word in cleaned_data:
                if directory_path == 'pos':
                    return_dict[(word, 1)] += 1
                else:
                    return_dict[(word, 0)] += 1
    return return_dict


def calculate_vocabulary(positive_review='pos', negative_review='neg'):
    vocabulary = set()
    directories = [positive_review, negative_review]
    for directory in directories:
        for item in os.listdir(directory):
            with open(os.path.join(directory, item), 'rt') as item_file:
                item_next = item_file.read()
                cleaned_data = process_data(item_next)
                for word in cleaned_data:
                    vocabulary.add(word)

    return len(vocabulary)


def train_naiveBayes(positive_dict, negative_dict, reviews):
    loglikelihoods = {}
    total_document = len(os.listdir("pos")) + len(os.listdir("neg"))
    Dpos = len(os.listdir("pos"))
    Dneg = len(os.listdir("neg"))
    Npos = len(positive_dict)
    Nneg = len(negative_dict)
    prior_dpos = Dpos / total_document
    prior_dneg = Dneg / total_document
    V = calculate_vocabulary()
    logprior = math.log(prior_dpos) - math.log(prior_dneg)

    with open(os.path.join(reviews), 'rt') as file:
        cleaned_data = process_data(file)
        for word in cleaned_data:
            freq_pos = positive_dict[word]
            freq_neg = negative_dict[word]
            proba_wpos = (freq_pos + 1) / (Npos + V)
            proba_wneg = (freq_neg + 1) / (Nneg + V)
            loglikelihood = math.log(proba_wpos / proba_wneg)
            loglikelihoods[word] = loglikelihood
    return logprior, loglikelihoods


def main():
    # cleaned_data = process_data("This is a great movie !")
    # print(cleaned_data)
    # happy_dict = count_reviews("pos")
    # negative_dict = count_reviews("neg")
    # print(happy_dict)
    # print(negative_dict)


if __name__ == '__main__':
    main()
