import math
import os
import random

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
                for word in item_next:
                    vocabulary.add(word)

    return len(vocabulary)


def train_naiveBayes(positive_dict, negative_dict, reviews):
    loglikelihoods = defaultdict(float)
    total_document = len(os.listdir("pos")) + len(os.listdir("neg"))
    Dpos = len(os.listdir("pos"))
    Dneg = len(os.listdir("neg"))
    Npos = len(positive_dict)
    Nneg = len(negative_dict)
    prior_dpos = Dpos / total_document
    prior_dneg = Dneg / total_document
    V = calculate_vocabulary()
    logprior = math.log(prior_dpos) - math.log(prior_dneg)

    for review in reviews:
        directory = "pos" if review in os.listdir("pos") else "neg"
        with open(os.path.join(directory, review), 'rt') as file:
            file_content = file.read()
            cleaned_data = process_data(file_content)
            for word in cleaned_data:
                freq_pos = positive_dict[(word, 1)]
                freq_neg = negative_dict[(word, 0)]
                proba_wpos = (freq_pos + 1) / (Npos + V)
                proba_wneg = (freq_neg + 1) / (Nneg + V)
                loglikelihood = math.log(proba_wpos / proba_wneg)
                loglikelihoods[word] = loglikelihood

    return logprior, loglikelihoods


def predict_naiveBayes(review_text, logprior, loglikelihoods):
    cleaned_data = process_data(review_text)
    p = logprior
    for word in cleaned_data:
        p += loglikelihoods[word]
    return p


def pick_random_sample():
    positive_review = os.listdir("pos")
    negative_review = os.listdir("neg")
    total_reviews = positive_review + negative_review
    random.shuffle(total_reviews)
    split_index = int(len(total_reviews) * 0.8)
    train_reviews = total_reviews[:split_index]
    test_reviews = total_reviews[split_index:]
    return train_reviews, test_reviews


def test_naiveBayes(test_reviews, logprior, loglikelihoods):
    positive_reviews = os.listdir("pos")
    negative_reviews = os.listdir("neg")
    result_dict = defaultdict(int)
    for review in test_reviews:
        directory = "pos" if review in os.listdir("pos") else "neg"
        with open(os.path.join(directory, review), 'r') as file:
            review_text = file.read()
        pred_res = predict_naiveBayes(review_text, logprior, loglikelihoods)
        result_dict[review] = 1 if pred_res > 0 else 0
    correct_pred = 0
    for review, prediction in result_dict.items():
        if review in positive_reviews and prediction == 1:
            correct_pred += 1
        elif review in negative_reviews and prediction == 0:
            correct_pred += 1
    return correct_pred / len(test_reviews), result_dict


def error_analysis(result_dict):
    positive_reviews = os.listdir("pos")
    negative_reviews = os.listdir("neg")

    with open("error_analysis.txt", 'w') as file:
        file.write(f"{'Truth':<10} {'Predicted':<10} {'Review':<50}\n")
        for review, prediction in result_dict.items():
            if (review in positive_reviews and prediction == 0) or (review in negative_reviews and prediction == 1):
                truth = 1 if review in positive_reviews else 0
                with open(os.path.join("pos" if truth == 1 else "neg", review), 'r') as review_file:
                    review_text = review_file.read().strip()
                    review_words = review_text.split()
                    if len(review_words) > 20:
                        review_text = " ".join(review_words[:20]) + "..."
                file.write(f"{truth:<10} {prediction:<10} {review_text:<50}\n")


def predict_lionking_reviews(logprior, loglikelihoods):
    with open("LionKing_MovieReviews.txt", 'r') as input_file, open("LionKing_Output.txt", 'w') as output_file:
        for review in input_file:
            review_text = review.strip()
            sentiment_score = predict_naiveBayes(review_text, logprior, loglikelihoods)
            sentiment = 1 if sentiment_score > 0 else 0
            output_file.write(f"{review_text} | Sentiment: {sentiment}\n")


def main():
    positive_dict = count_reviews("pos")
    negative_dict = count_reviews("neg")
    train_reviews, test_reviews = pick_random_sample()
    print('Training Naive Bayes Model...')
    logprior, loglikelihoods = train_naiveBayes(positive_dict, negative_dict, train_reviews)
    print('Training done.')
    print('Testing Naive Bayes Model....')
    model_accuracy, result_dict = test_naiveBayes(test_reviews, logprior, loglikelihoods)
    print('Testing done')
    print(f"Naive Bayes accuracy = {model_accuracy:.4f}")

    error_analysis(result_dict)

    predict_lionking_reviews(logprior, loglikelihoods)


if __name__ == '__main__':
    main()
