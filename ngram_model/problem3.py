import math
from collections import defaultdict
from problem1 import read_files, split_data, tokenize
from problem2 import create_bigram_dict


def smoothed_bigram_model(bigram_dict, word_dict, output_file, delta=0.1):
    with open(output_file, 'w') as file:
        for (prev_word, word), count in bigram_dict.items():
            smoothed_count = count + delta
            smoothed_total_count = word_dict[prev_word] + (delta * len(word_dict))
            probability = smoothed_count / smoothed_total_count
            file.write(f"P({word} | {prev_word}) {probability}\n")


def evaluate_smoothed_bigram_model(bigram_dict, word_dict, test_sentences, output_file, delta=0.1):
    with open(output_file, 'w') as file:
        for sentence in test_sentences:
            words = sentence.lower().split()
            sent_prob = 1.0
            prev_word = None
            for word in words:
                if prev_word is not None:
                    bigram = (prev_word, word)
                    smoothed_count = bigram_dict[bigram] + delta if bigram in bigram_dict else delta
                    smoothed_total_count = word_dict[prev_word] + (delta * len(word_dict))
                    probability = smoothed_count / smoothed_total_count
                    sent_prob *= probability
                prev_word = word
            file.write(f"p({sentence.strip()}) = {sent_prob}\n")


def calculate_perplexity(prob_file, N):
    with open(prob_file, 'r') as file:
        probs = [float(line.split('=')[-1].strip()) for line in file]

    # Handle zero probabilities
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
    bigram_dict, word_dict = create_bigram_dict(train_sentences)
    smoothed_bigram_model(bigram_dict, word_dict, "smooth_probs.txt", delta=0.1)
    evaluate_smoothed_bigram_model(bigram_dict, word_dict, test_sentences, 'smoothed_eval.txt', delta=0.1)
    perplexity = calculate_perplexity('smoothed_eval.txt', len(test_sentences))
    print(f"Smoothed Bigram Model Perplexity: {perplexity}")


if __name__ == "__main__":
    main()
