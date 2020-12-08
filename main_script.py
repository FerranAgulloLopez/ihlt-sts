from scipy.stats import pearsonr
import matplotlib.pyplot as plt

from auxiliary_code.file_methods import load_train_data, load_test_data
from auxiliary_code.preprocessing_steps import Preprocessing
from auxiliary_code.other_methods import pretty_print_sentence
from auxiliary_code.sentence_similarity_metrics import SentenceSimilarity
from auxiliary_code.aggregation_methods import run_aggregation_method

train_sentence_pairs, train_labels = load_train_data()
test_sentence_pairs, test_labels = load_test_data()

print('Train values length:', len(train_sentence_pairs), '; Train labels length:', len(train_labels))
print('Test values length:', len(test_sentence_pairs), '; Test labels length:', len(test_labels))

config = {
    'preprocessing_steps': [
        {'name': 'lower_case'},
        {'name': 'word_tokenize'},
        {'name': 'punctuation_removal'},
        {'name': 'pos_tagging'},
        {'name': 'lemmatization'},
        {'name': 'word_sense_disambiguation'}
    ],
    'similarity_metrics': [
        {'name': 'jaccard_similarity'},
        {'name': 'ngram_overlap', 'n': 2, 'filter': 'none'},
        {'name': 'ngram_overlap', 'n': 3, 'filter': 'none'},
        {'name': 'ngram_overlap', 'n': 1, 'filter': 'content'},
        {'name': 'ngram_overlap', 'n': 2, 'filter': 'content'},
        {'name': 'ngram_overlap', 'n': 3, 'filter': 'content'},
        {'name': 'ngram_overlap', 'n': 4, 'filter': 'content'},
        {'name': 'ngram_overlap', 'n': 1, 'filter': 'stopwords'},
        {'name': 'ngram_overlap', 'n': 2, 'filter': 'stopwords'},
        {'name': 'ngram_overlap', 'n': 3, 'filter': 'stopwords'},
        {'name': 'ngram_overlap', 'n': 4, 'filter': 'stopwords'},
        {'name': 'ngram_overlap', 'n': 5, 'filter': 'stopwords'},
        {'name': 'pos_ngram_overlap', 'n': 1},
        {'name': 'pos_ngram_overlap', 'n': 2},
        {'name': 'character_ngram_overlap', 'n': 2},
        {'name': 'character_ngram_overlap', 'n': 3},
        {'name': 'character_ngram_overlap', 'n': 4},
        {'name': 'character_ngram_overlap', 'n': 5},
        {'name': 'character_ngram_overlap', 'n': 6},
        {'name': 'character_ngram_overlap', 'n': 7},
        {'name': 'character_ngram_overlap', 'n': 8},
        {'name': 'character_ngram_overlap', 'n': 9},
        {'name': 'sentence_length_difference'},
        {'name': 'wordnet_pairwise_word_similarity', 'metric': 'lch'},
        {'name': 'wordnet_pairwise_word_similarity', 'metric': 'path'},
        # {'name': 'wordnet_pairwise_word_similarity', 'metric': 'wup'},
        {'name': 'number_overlap'},
        {'name': 'dependency_overlap', 'content': False},
        # {'name': 'longest_common_subsequence', 'mode': 'subsequence'},
        {'name': 'longest_common_subsequence', 'mode': 'substring'}
    ],
    'aggregation': {'name': 'krr'}
}

preprocessing = Preprocessing(config['preprocessing_steps'])
train_output = preprocessing.do_pipeline(train_sentence_pairs)
test_output = preprocessing.do_pipeline(test_sentence_pairs)

print('\nSentence 1')
pretty_print_sentence(train_output[0][0])
print('\nSentence 2')
pretty_print_sentence(train_output[0][1])

sentence_similarity = SentenceSimilarity(config['similarity_metrics'])
train_results = sentence_similarity.compute_pair_comparison(train_output)
test_results = sentence_similarity.compute_pair_comparison(test_output)

train_results = run_aggregation_method(config['aggregation'], train_results, train_labels)
test_results = run_aggregation_method(config['aggregation'], test_results, train_labels, test=True)

print('Train results')
print(train_results)
print('\nTest results')
print(test_results)

def show_scatter_plot(labels, results, title):
    plt.figure()
    plt.scatter(labels, results)
    plt.xlim([0, 5])
    plt.ylim([0, 5])
    plt.xlabel('Gold standard')
    plt.ylabel('Similarity')
    plt.title(title)
    plt.show()

print('Train results')
print(pearsonr(train_labels, train_results)[0])
print('\nTest results')
print(pearsonr(test_labels, test_results)[0])

show_scatter_plot(train_labels, train_results, 'Similarity vs Gold standard in the training set')
show_scatter_plot(test_labels, test_results, 'Similarity vs Gold standard in the testing set')
