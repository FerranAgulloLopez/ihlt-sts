from nltk.metrics import jaccard_distance
from pycorenlp import StanfordCoreNLP
from nltk.corpus.reader.wordnet import Synset
from nltk.corpus import stopwords
from nltk import word_tokenize
from nltk.parse.corenlp import CoreNLPDependencyParser
from nltk.stem import WordNetLemmatizer
import pylcs
import numpy as np
from word2number import w2n
import math

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
        metric_names = {x['name'] for x in self.metrics}
        if 'wordnet_pairwise_word_similarity_weighted' in metric_names or 'ngram_overlap_weighted' in metric_names:
            self.word_idfs = self._compute_word_idfs(sentence_pairs)

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

    def _filter_stopwords(self, sentence):
        stopword_list = stopwords.words('english')
        return [word
                for word in sentence[2]
                if word in stopword_list]

    def _idf_process_sentence(self, sentence, word_apparitions):
        found_words = {}
        for word in sentence[2]:
            if word not in word_apparitions:
                word_apparitions[word] = 1
                found_words[word] = True
            elif word not in found_words:
                word_apparitions[word] += 1
                found_words[word] = True
        return word_apparitions

    def _compute_word_idfs(self, sentence_pairs):
        # TODO can be done in O(n), better than O(2n)

        # Compute word apparitions in the sentences
        word_apparitions = {}
        for sentence1, sentence2 in sentence_pairs:
            word_apparitions = self._idf_process_sentence(sentence1, word_apparitions)
            word_apparitions = self._idf_process_sentence(sentence2, word_apparitions)

        # Compute idf
        word_idfs = {}
        number_sentences = len(sentence_pairs)*2
        for word, apparitions in word_apparitions.items():
            word_idfs[word] = math.log(number_sentences/float(apparitions))
        return word_idfs

    def jaccard_similarity(self, config, sentence1, sentence2):
        return 1 - jaccard_distance(set(sentence1[2]), set(sentence2[2]))

    def ngram_overlap(self, config, sentence1, sentence2):
        n = config['n'] if 'n' in config else 1
        filter = config['filter'] if 'filter' in config else 'none'

        if filter == 'content':
            s1 = self._filter_content_words(sentence1)
            s2 = self._filter_content_words(sentence2)
        elif filter == 'stopwords':
            s1 = self._filter_stopwords(sentence1)
            s2 = self._filter_stopwords(sentence2)
        else:  # filter == 'none'
            s1 = sentence1[2]
            s2 = sentence2[2]

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

    def ngram_overlap_weighted(self, config, sentence1, sentence2):
        # TODO wrap inside the previous function
        n = config['n'] if 'n' in config else 1
        filter = config['filter'] if 'filter' in config else 'none'

        if filter == 'content':
            s1 = self._filter_content_words(sentence1)
            s2 = self._filter_content_words(sentence2)
        elif filter == 'stopwords':
            s1 = self._filter_stopwords(sentence1)
            s2 = self._filter_stopwords(sentence2)
        else:  # filter == 'none'
            s1 = sentence1[2]
            s2 = sentence2[2]

        set1 = set()
        set2 = set()
        for i in range(len(s1)-n+1):
            set1.add(tuple(s1[i:i+n]))
        for i in range(len(s2)-n+1):
            set2.add(tuple(s2[i:i+n]))

        intersection = set1.intersection(set2)
        mean_idf = np.mean(np.asarray([self.word_idfs[word[0]] for word in intersection])) if len(intersection) > 0 else 0
        sizei = len(intersection)
        size1, size2 = len(set1), len(set2)
        try:
            return (2 * (1 / (size1 / sizei + size2 / sizei)))*mean_idf # TODO find other way, to not destroy normalization between 0 and 1
        except ZeroDivisionError:
            return 0

    def character_ngram_overlap(self, config, sentence1, sentence2):
        n = config['n'] if 'n' in config else 1

        s1 = sentence1[1]
        s2 = sentence2[1]

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

    def pos_ngram_overlap(self, config, sentence1, sentence2):
        n = config['n'] if 'n' in config else 1

        s1 = sentence1[3]
        s2 = sentence2[3]

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
        lch = lambda w1, w2: w1.lch_similarity(w2)
        path = lambda w1, w2: w1.path_similarity(w2)
        wup = lambda w1, w2: w1.wup_similarity(w2)
        metrics = {'lch': lch, 'path': path, 'wup': wup}
        metric = config['metric'] if 'metric' in config else 'lch'

        def _compute_sentence_similarities(s1, s2):
            similarities = []
            for word in s1[4]:
                if not isinstance(word, Synset):
                    continue
                max_similarity = 0
                for other_word in s2[4]:
                    if isinstance(other_word, Synset) and word.pos() == other_word.pos():
                        similarity = metrics[metric](word, other_word)
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

    def wordnet_pairwise_word_similarity_weighted(self, config, sentence1, sentence2):
        # TODO wrap this function inside the prior one
        lch = lambda w1, w2: w1.lch_similarity(w2)
        path = lambda w1, w2: w1.path_similarity(w2)
        wup = lambda w1, w2: w1.wup_similarity(w2)
        metrics = {'lch': lch, 'path': path, 'wup': wup}
        metric = config['metric'] if 'metric' in config else 'lch'

        def _compute_sentence_similarities(s1, s2):
            similarities = []
            for index, word in enumerate(s1[4]):
                if not isinstance(word, Synset):
                    continue
                max_similarity = 0
                for other_word in s2[4]:
                    if isinstance(other_word, Synset) and word.pos() == other_word.pos():
                        similarity = metrics[metric](word, other_word)
                        if similarity is not None:
                            max_similarity = max(similarity, max_similarity)
                similarities.append(max_similarity*self.word_idfs[s1[2][index]])
            return similarities

        scores1 = _compute_sentence_similarities(sentence1, sentence2)
        scores2 = _compute_sentence_similarities(sentence2, sentence1)
        if scores1 == [] or scores2 == []:
            return 0
        score1, score2 = np.mean(scores1), np.mean(scores2)
        return np.mean([score1, score2])

    def number_overlap(self, config, sentence1, sentence2):
        numbers1 = []
        for i, word in enumerate(sentence1[2]):
            if sentence1[3][i] == 'CD':
                try:
                    numbers1.append(w2n.word_to_num(sentence1[2][i]))
                except ValueError:
                    try:
                        numbers1.append(w2n.word_to_num(sentence1[2][i].replace(',', '')))
                    except ValueError:
                        try:
                            numbers1.append(float(sentence1[2][i].replace(',', '')))
                        except ValueError:
                            numbers1.append(word)
        numbers2 = []
        for i, word in enumerate(sentence2[2]):
            if sentence2[3][i] == 'CD':
                try:
                    numbers2.append(w2n.word_to_num(sentence2[2][i]))
                except ValueError:
                    try:
                        numbers2.append(w2n.word_to_num(sentence2[2][i].replace(',', '')))
                    except ValueError:
                        try:
                            numbers2.append(float(sentence2[2][i].replace(',', '')))
                        except ValueError:
                            numbers2.append(word)
        try:
            return 1 - jaccard_distance(set(numbers1), set(numbers2))
        except ZeroDivisionError:
            return 0

    def dependency_overlap(self, config, sentence1, sentence2):
        content = config['content'] if 'content' in config else False

        dep = CoreNLPDependencyParser('http://localhost:9000')
        triples1 = list(list(dep.parse(sentence1[2]))[0].triples())
        triples2 = list(list(dep.parse(sentence2[2]))[0].triples())
        lm = WordNetLemmatizer()
        if content:
            triples1 = [(src, rel, dest)
                        for src, rel, dest in triples1
                        if src[1][0] in ['V', 'N', 'R', 'J'] and dest[1][0] in ['V', 'N', 'R', 'J']]
            triples2 = [(src, rel, dest)
                        for src, rel, dest in triples2
                        if src[1][0] in ['V', 'N', 'R', 'J'] and dest[1][0] in ['V', 'N', 'R', 'J']]

        triples1 = [(lm.lemmatize(src[0]), rel, lm.lemmatize(dest[0])) for src, rel, dest in triples1]
        triples2 = [(lm.lemmatize(src[0]), rel, lm.lemmatize(dest[0])) for src, rel, dest in triples2]

        set1 = set(triples1)
        set2 = set(triples2)
        intersection = set1.intersection(set2)
        sizei = len(intersection)
        size1, size2 = len(set1), len(set2)
        try:
            return 2 * (1 / (size1 / sizei + size2 / sizei))
        except ZeroDivisionError:
            return 0

    def longest_common_subsequence(self, config, sentence1, sentence2):
        mode = config['mode'] if 'mode' in config else 'subsequence'
        if mode == 'subsequence':
            return pylcs.lcs(sentence1[1], sentence2[1])
        else:  # mode == 'substring'
            return pylcs.lcs2(sentence1[1], sentence2[1])



