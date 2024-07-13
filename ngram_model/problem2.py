import math
from collections import defaultdict
from problem1 import read_files, split_data


def create_bigram_dict(sentences):
    bigram_dict = defaultdict(int)
    word_dict = defaultdict(int)
    for sentence in sentences:
        words = sentence.lower().split()
        prev_word = None
        for word in words:
            if prev_word is not None:
                bigram = (prev_word, word)
                bigram_dict[bigram] += 1
            word_dict[word] += 1
            prev_word = word
    return bigram_dict, word_dict


def build_bigram_model(bigram_dict, word_dict, output_file):
    with open(output_file, 'w') as file:
        for (prev_word, word), count in bigram_dict.items():
            probability = count / word_dict[prev_word]
            file.write(f"P({word} | {prev_word}) {probability}\n")


def evaluate_bigram_model(bigram_dict, word_dict, test_sentences, output_file):
    with open(output_file, 'w') as file:
        for sentence in test_sentences:
            words = sentence.lower().split()
            sent_prob = 1.0
            prev_word = None
            for word in words:
                if prev_word is not None:
                    bigram = (prev_word, word)
                    if bigram in bigram_dict:
                        probability = bigram_dict[bigram] / word_dict[prev_word]
                        sent_prob *= probability
                    else:
                        sent_prob *= 0
                prev_word = word
            file.write(f"{sent_prob}\n")


def calculate_perplexity(prob_file, N):
    with open(prob_file, 'r') as file:
        probs = [float(line.strip()) for line in file]

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
    build_bigram_model(bigram_dict, word_dict, "bigram_probs.txt")
    evaluate_bigram_model(bigram_dict, word_dict, test_sentences, 'bigram_eval.txt')
    perplexity = calculate_perplexity('bigram_eval.txt', len(test_sentences))
    print(f"Bigram Model Perplexity: {perplexity}")


if __name__ == "__main__":
    main()
