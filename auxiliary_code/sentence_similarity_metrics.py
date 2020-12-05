from nltk.metrics import jaccard_distance
import numpy as np

# Sentence data saved inside an array -> 0: original sentence; 1: sentence transformations; 2: tokens; 3: pos tags;  4: synsets


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

    def jaccard_similarity(self, config, sentence1, sentence2):
        return 1 - jaccard_distance(set(sentence1[2]),set(sentence2[2]))
