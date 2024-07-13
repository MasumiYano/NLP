from problem2 import *


def smoothed_bigram_model(bigram_dict, word_dict, output_file, delta=0.01):
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
            previous_word = None
            for word in words:
                if previous_word is not None:
                    bigram = (previous_word, word)
                    smoothed_count = bigram_dict[bigram] + delta if bigram in bigram_dict else delta
                    smoothed_total_count = word_dict[previous_word] + (delta * len(word_dict))
                    probability = smoothed_count / smoothed_total_count
                    sent_prob *= probability
                previous_word = word
            file.write(f"{sent_prob}\n")


def calculate_perplexity(prob_file, N):
    with open(prob_file, 'r') as file:
        probs = [float(line.strip()) for line in file]
    product = math.prod(probs)
    perplexity = product ** (-1 / N)
    return perplexity


def main():
    file_paths = ['doyle_Bohemia.txt']
    sentences = read_files(file_paths)
    train_sentences, test_sentences = split_data(sentences)
    bigram_dict, word_dict = create_bigram_dict(train_sentences)
    smoothed_bigram_model(bigram_dict, word_dict, "smooth_prob.txt")
    evaluate_bigram_model(bigram_dict, word_dict, test_sentences, 'smooth_eval.txt')
    perplexity = calculate_perplexity('smooth_eval.txt', len(test_sentences))
    print(f"Smoothed Bigram Model Perplexity: {perplexity}")


if __name__ == "__main__":
    main()