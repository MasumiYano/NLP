import math
import random
from collections import defaultdict


def read_files(file_paths):
    sentences = []
    for file_path in file_paths:
        with open(file_path, 'r') as file:
            sentences.extend(file.readlines())
    return sentences


def split_data(sentences, train_ratio=0.8):
    split_index = int(len(sentences) * train_ratio)
    return sentences[:split_index], sentences[split_index:]


def create_word_dictionary(sentences):
    word_freq = defaultdict(int)
    for sentence in sentences:
        words = sentence.lower().split()
        for word in words:
            word_freq[word] += 1
    return word_freq


def build_unigram_model(word_dict, output_file):
    total_count = sum(word_dict.values())
    with open(output_file, 'w') as file:
        for word, count in word_dict.items():
            probability = count / total_count
            file.write(f"P({word}) {probability}\n")


def evaluate_unigram_model(word_dict, test_sentences, output_file):
    total_count = sum(word_dict.values())
    word_probs = {word: count / total_count for word, count in word_dict.items()}

    with open(output_file, 'w') as file:
        for sentence in test_sentences:
            words = sentence.lower().split()
            sent_prob = 1.0
            for word in words:
                if word in word_probs:
                    sent_prob *= word_probs[word]
                else:
                    sent_prob *= 0
            file.write(f"{sent_prob}\n")


def calculate_perplexity(prob_file, N):
    with open(prob_file, 'r') as file:
        probs = [float(line.strip()) for line in file]

    # zero probabilities
    non_zero_probs = [p for p in probs if p > 0]
    if not non_zero_probs:
        return float('inf')

    product = math.prod(non_zero_probs)
    perplexity = product ** (-1 / N)
    return perplexity


def main():
    file_paths = ['doyle_Bohemia.txt']
    sentences = read_files(file_paths)
    train_sentences, test_sentences = split_data(sentences)
    word_dict = create_word_dictionary(train_sentences)
    build_unigram_model(word_dict, 'unigram_probs.txt')
    evaluate_unigram_model(word_dict, test_sentences, 'unigram_eval.txt')
    perplexity = calculate_perplexity('unigram_eval.txt', len(test_sentences))
    print(f"Unigram Model Perplexity: {perplexity}")


if __name__ == "__main__":
    main()
