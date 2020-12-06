from nltk.metrics import jaccard_distance
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

    def jaccard_similarity(self, config, sentence1, sentence2):
        return 1 - jaccard_distance(set(sentence1[2]), set(sentence2[2]))

    def ngram_overlap(self, config, sentence1, sentence2):
        n = config['n'] if 'n' in config else 1
        content = config['content'] if 'content' in config else False

        s1 = sentence1[2]
        s2 = sentence2[2]

        if content:
            s1 = [sentence1[2][i]
                  for i in range(len(sentence1[3]))
                  if sentence1[3][i][0] in ['V', 'N', 'R', 'J']]
            s2 = [sentence2[2][i]
                  for i in range(len(sentence2[3]))
                  if sentence2[3][i][0] in ['V', 'N', 'R', 'J']]

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



