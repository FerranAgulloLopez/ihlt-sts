from nltk.metrics import jaccard_distance
from nltk.corpus import wordnet as wn
from nltk.corpus.reader.wordnet import Synset
from nltk.wsd import lesk
import numpy as np

# Sentence data saved inside an array:
#     0: original sentence
#     1: sentence transformations
#     2: tokens
#     3: pos tags
#     4: synsets


class SentenceSimilarity:

    def __init__(self, config):
        self.metrics = config

    # Main methods

    def compute_pair_comparison(self, sentence_pairs):
        output = np.zeros((len(sentence_pairs), len(self.metrics)))
        for index, pair in enumerate(sentence_pairs):
            output[index] = self.run_sentence_similarity_metrics(self.metrics, pair[0], pair[1])
            index += 1
        return output

    def run_sentence_similarity_metrics(self, metrics, sentence1, sentence2):
        output = np.zeros(len(metrics))
        for index, metric in enumerate(metrics):
            output[index] = eval('self.' + metric['name'])(metric, sentence1, sentence2)
        return output

    # Auxiliary methods

    def _filter_content_words(self, sentence):
        return [sentence[2][i]
                for i in range(len(sentence[3]))
                if sentence[3][i][0] in ['V', 'N', 'R', 'J']]

    def jaccard_similarity(self, config, sentence1, sentence2):
        return 1 - jaccard_distance(set(sentence1[2]), set(sentence2[2]))

    def ngram_overlap(self, config, sentence1, sentence2):
        n = config['n'] if 'n' in config else 1
        content = config['content'] if 'content' in config else False

        s1 = sentence1[2]
        s2 = sentence2[2]

        if content:
            s1 = self._filter_content_words(sentence1)
            s2 = self._filter_content_words(sentence2)

        set1 = set()
        set2 = set()
        for i in range(len(s1)-n+1):
            set1.add(tuple(s1[i:i+n]))
        for i in range(len(s2)-n+1):
            set2.add(tuple(s2[i:i+n]))

        intersection = set1.intersection(set2)
        sizei = len(intersection)
        size1, size2 = len(set1), len(set2)
        try:
            return 2 * (1 / (size1 / sizei + size2 / sizei))
        except ZeroDivisionError:
            return 0

    def sentence_length_difference(self, config, sentence1, sentence2):
        return np.abs(len(sentence1[2]) - len(sentence2[2]))

    def wordnet_pairwise_word_similarity(self, config, sentence1, sentence2):
        def _compute_sentence_similarities(s1, s2):
            similarities = []
            for word in s1[4]:
                if not isinstance(word, Synset):
                    continue
                max_similarity = 0
                for other_word in s2[4]:
                    if isinstance(other_word, Synset) and word.pos() == other_word.pos():
                        similarity = word.wup_similarity(other_word)
                        if similarity is not None:
                            max_similarity = max(similarity, max_similarity)
                similarities.append(max_similarity)
            return similarities

        scores1 = _compute_sentence_similarities(sentence1, sentence2)
        scores2 = _compute_sentence_similarities(sentence2, sentence1)
        if scores1 == [] or scores2 == []:
            return 0
        score1, score2 = np.mean(scores1), np.mean(scores2)
        return np.mean([score1, score2])



