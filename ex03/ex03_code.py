import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np
import re
import os
from wordcloud import WordCloud

# Ensure necessary NLTK data is downloaded
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')


def load_text(file_path):
    with open(file_path, 'r') as file:
        return file.read()


def tokenize_text(text):
    return word_tokenize(text)


def filter_tokens(tokens):
    return [token.lower() for token in tokens if token.isalpha()]


def count_frequencies(tokens):
    return Counter(tokens)


def plot_frequencies(counter, title, output_path):
    tokens, frequencies = zip(*counter.items())
    ranks = np.arange(1, len(frequencies) + 1)
    plt.figure(figsize=(10, 6))
    plt.plot(np.log(ranks), np.log(frequencies), marker='o', linestyle='none')
    plt.title(title)
    plt.xlabel('Rank')
    plt.ylabel('Frequency')
    plt.savefig(output_path)
    plt.close()


def remove_stopwords(tokens):
    stop_words = set(stopwords.words('english'))
    return [token for token in tokens if token.lower() not in stop_words]


def stem_tokens(tokens):
    stemmer = PorterStemmer()
    return [stemmer.stem(token) for token in tokens]


def pos_tagging(tokens):
    return nltk.pos_tag(tokens)


def extract_adj_noun_phrases(pos_tags):
    phrases = []
    phrase = []
    for word, pos in pos_tags:
        if word.isalpha() and word.lower() not in {'s', 'second', 'servant', 'first', 'third'}:  # Words that kept
            # showing up and cause problems, so we put them out manually.
            if pos.startswith('JJ'):
                phrase.append(word)
            elif pos.startswith('NN') and len(phrase) > 0:
                phrase.append(word)
                if len(phrase) > 1:
                    phrases.append(' '.join(phrase))
                phrase = []
            else:
                phrase = []
        else:
            phrase = []
    return phrases


def find_homographs(pos_tags):
    pos_dict = {}
    for word, pos in pos_tags:
        word_lower = word.lower()
        if word_lower.isalpha() and word_lower:
            if word_lower not in pos_dict:
                pos_dict[word_lower] = set()
            pos_dict[word_lower].add(pos)
    return {word: list(pos) for word, pos in pos_dict.items() if len(pos) > 1}


def find_repeated_words(text):
    pattern = r'\b(\w+)\b(?:\W+\b\1\b)'
    matches = re.findall(pattern, text, re.IGNORECASE)
    return list(set(f"{match} {match}" for match in matches))


def create_tag_cloud(proper_nouns, output_path):
    wordcloud = WordCloud(width=800, height=400).generate(' '.join(proper_nouns))
    plt.figure(figsize=(10, 6))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.savefig(output_path)
    plt.close()


def save_results_to_file(output_dir, filename, data):
    with open(os.path.join(output_dir, filename), 'w') as f:
        for item in data:
            f.write(f"{item}\n")


def main():
    output_dir = './output'
    os.makedirs(output_dir, exist_ok=True)

    text = load_text('Romeo & Juliet.txt')

    tokens = tokenize_text(text)
    filtered_tokens = filter_tokens(tokens)

    # Part (b)
    token_freqs = count_frequencies(filtered_tokens)
    plot_frequencies(token_freqs, "Token Frequencies", os.path.join(output_dir, 'plot_b.png'))
    save_results_to_file(output_dir, 'answer_b.txt',
                         [f"{token}: {freq}" for token, freq in token_freqs.most_common(20)])

    # Part (c)
    tokens_no_stop = remove_stopwords(filtered_tokens)
    token_freqs_no_stop = count_frequencies(tokens_no_stop)
    plot_frequencies(token_freqs_no_stop, "Token Frequencies (Without Stopwords)",
                     os.path.join(output_dir, 'plot_c.png'))
    save_results_to_file(output_dir, 'answer_c.txt',
                         [f"{token}: {freq}" for token, freq in token_freqs_no_stop.most_common(20)])

    # Part (d)
    stemmed_tokens = stem_tokens(tokens_no_stop)
    token_freqs_stemmed = count_frequencies(stemmed_tokens)
    plot_frequencies(token_freqs_stemmed, "Stemmed Token Frequencies", os.path.join(output_dir, 'plot_d.png'))
    save_results_to_file(output_dir, 'answer_d.txt',
                         [f"{token}: {freq}" for token, freq in token_freqs_stemmed.most_common(20)])

    # Part (e)
    pos_tags = pos_tagging(tokens)
    adj_noun_phrases = extract_adj_noun_phrases(pos_tags)
    adj_noun_freqs = count_frequencies(adj_noun_phrases)
    plot_frequencies(adj_noun_freqs, "Adj+Noun Phrase Frequencies", os.path.join(output_dir, 'plot_e.png'))
    save_results_to_file(output_dir, 'answer_e.txt',
                         [f"{phrase}: {freq}" for phrase, freq in adj_noun_freqs.most_common(20)])

    # Part (g)
    homographs = find_homographs(pos_tags)
    sorted_homographs = sorted(homographs.items(), key=lambda item: len(item[1]), reverse=True)
    save_results_to_file(output_dir, 'answer_g.txt', [f"{word}: {pos}" for word, pos in sorted_homographs[:10]])
    save_results_to_file(output_dir, 'answer_g.txt', [f"{word}: {pos}" for word, pos in sorted_homographs[-10:]])

    # Part (h)
    proper_nouns = [word for word, pos in pos_tags if pos in ['NNP', 'NNPS']]
    create_tag_cloud(proper_nouns, os.path.join(output_dir, 'tag_cloud_h.png'))

    # Part (i)
    repeated_words = find_repeated_words(text)
    save_results_to_file(output_dir, 'answer_i.txt', repeated_words)


if __name__ == "__main__":
    main()
