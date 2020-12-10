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
from nltk.corpus import wordnet as wn
from nltk.stem import PorterStemmer
import string
from scipy import spatial

# Sentence data saved inside an array:
#     0: original sentence
#     1: sentence transformations
#     2: tokens
#     3: pos tags
#     4: synsets


class SentenceSimilarity:

    def __init__(self, config):
        self.metrics = config
        self.index = 0 # TODO refactor, necessary for cosine similarity

    # Main methods

    def compute_pair_comparison(self, sentence_pairs):
        """
        Computes, for every sentence pair in the list sentence_pairs, all metrics in the configuration.
        If any metric requires it, the Tf-idf are computed.
        """
        metric_names = {x['name'] for x in self.metrics}
        if 'wordnet_pairwise_word_similarity_weighted' in metric_names or 'ngram_overlap_weighted' in metric_names or 'cosine_similarity_tfidf' in metric_names:
            self.word_idfs, self.sentence_tfidfs = self._compute_word_tfidfs(sentence_pairs)

        if 'jaccard_similarity_stemmer' in metric_names:
            self.stemmer = PorterStemmer()
            self.punctuation_set = set(string.punctuation)
            self.punctuation_set.add('``')
            self.punctuation_set.add('\'\'')

        output = np.zeros((len(sentence_pairs), len(self.metrics)))
        for index, pair in enumerate(sentence_pairs):
            self.index = index
            output[index] = self.run_sentence_similarity_metrics(self.metrics, pair[0], pair[1])
        return output

    def run_sentence_similarity_metrics(self, metrics, sentence1, sentence2):
        """
        Compute for a sentence pair all metrics in the configuration.
        """
        output = np.zeros(len(metrics))
        for index, metric in enumerate(metrics):
            output[index] = eval('self.' + metric['name'])(metric, sentence1, sentence2)
        return output

    # Auxiliary methods

    def _filter_content_words(self, sentence):
        """
        Filters all non-content words, leaving only verbs, nouns, adverbs and adjectives.
        """
        return [sentence[2][i]
                for i in range(len(sentence[3]))
                if sentence[3][i][0] in ['V', 'N', 'R', 'J']]

    def _filter_stopwords(self, sentence):
        """
        Leaves only the english stopwords in a sentence.
        """
        stopword_list = stopwords.words('english')
        return [word
                for word in sentence[2]
                if word in stopword_list]

    def _idf_process_sentence(self, sentence_id, sentence, corpus_word_apparitions, sentence_word_apparitions):
        """
        Computes document frequency for each word and term frequency for each word-sentence.
        """
        sentence_word_apparitions[sentence_id] = {}
        for word in sentence[2]:
            if word not in corpus_word_apparitions:
                corpus_word_apparitions[word] = 1
                sentence_word_apparitions[sentence_id][word] = 1
            elif word not in sentence_word_apparitions[sentence_id]:
                corpus_word_apparitions[word] += 1
                sentence_word_apparitions[sentence_id][word] = 1
            elif word in sentence_word_apparitions[sentence_id]:
                sentence_word_apparitions[sentence_id][word] += 1
        return corpus_word_apparitions, sentence_word_apparitions

    def _compute_word_tfidfs(self, sentence_pairs):
        """
        Computes Tf-idfs using all sentence pairs.
        """
        # Compute word apparitions in the sentences
        corpus_word_apparitions = {}
        sentence_word_apparitions = {}
        for index, (sentence1, sentence2) in enumerate(sentence_pairs):
            corpus_word_apparitions, sentence_word_apparitions = self._idf_process_sentence(str(index) + '1', sentence1, corpus_word_apparitions, sentence_word_apparitions)
            corpus_word_apparitions, sentence_word_apparitions = self._idf_process_sentence(str(index) + '2', sentence2, corpus_word_apparitions, sentence_word_apparitions)

        # Compute word idf and sentence tf-idf
        word_idfs = {}
        sentence_tfidfs = {}
        corpus_word_apparitions_ids = {word:index for index, word in enumerate(corpus_word_apparitions)}
        number_sentences = len(sentence_pairs)*2
        for id, words in sentence_word_apparitions.items():
            tfidfs = np.zeros(len(corpus_word_apparitions))
            for word, apparitions_in_sentence in words.items():
                apparitions_in_corpus = corpus_word_apparitions[word]
                if word not in word_idfs:
                    word_idfs[word] = math.log(number_sentences/(float(apparitions_in_corpus)+1))
                index = corpus_word_apparitions_ids[word]
                tfidfs[index] = apparitions_in_sentence*word_idfs[word]
            sentence_tfidfs[id] = tfidfs
        return word_idfs, sentence_tfidfs

    def jaccard_similarity(self, config, sentence1, sentence2):
        """
        Computes simple Jaccard similarity with the token lists of two sentences.
        Equivalent to word 1-gram similarity.
        """
        return 1 - jaccard_distance(set(sentence1[2]), set(sentence2[2]))

    def jaccard_similarity_stemmer(self, config, sentence1, sentence2):
        """
        Computes stem unigram similarity.
        """
        tokens1 = [self.stemmer.stem(token) for token in word_tokenize(sentence1[1]) if token not in self.punctuation_set]
        tokens2 = [self.stemmer.stem(token) for token in word_tokenize(sentence2[1]) if token not in self.punctuation_set]
        return 1 - jaccard_distance(set(tokens1), set(tokens2))

    def cosine_similarity_tfidf(self, config, sentence1, sentence2):
        """
        Computes the Tf-idf cosine distance between two sentences, using our own previously computed metric.
        """
        id1 = str(self.index)  + '1'
        id2 = str(self.index) + '2'
        return 1 - spatial.distance.cosine(self.sentence_tfidfs[id1], self.sentence_tfidfs[id2])

    def ngram_overlap(self, config, sentence1, sentence2):
        """
        Computes the content, stopword or word n-gram similarity using the list of tokens.
        Content: Using only the nouns, verbs, adverbs and adjectives of a sentence.
        Stopwords: Using only the stopwords of a sentence.
        Word: Using all tokens in a sentence, no filtering.
        """
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
        """
        Computes the n-gram similarities as ngram_overlap does, but adds a weighting factor using the Tf-idf.
        """
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
        """
        Computes the character n-gram similarities. In other words, it extracts n-grams character by character instead
        of word by word, using the original lowercased sentences.
        """
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
        """
        Computes the part of speech n-gram similarity using the previously computed PoS list of a sentence.
        """
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
        """
        Computes the absolute difference between the length in tokens of two sentences.
        """
        return np.abs(len(sentence1[2]) - len(sentence2[2]))

    def wordnet_pairwise_word_similarity(self, config, sentence1, sentence2):
        """
        Computes the Wordnet-based pairwise word similarity. This function uses the Wordnet synsets for each sentence,
        found with the Lesk algorithm in the preprocessing step.
        For each synset of a sentence, it computes the Path, Leacock-Chodrow or Wu-Palmer similarities (depending on
        configuration) with all synsets in the other sentence and stores the maximum similarity it founds. After storing
        all maximum similarities of the two sentences in a pair, it computes the mean similarity of each sentence and
        then the mean of the two sentence similarities.
        """
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
        """
        Computes the Wordnet-based pairwise word similarity as wordnet_pairwise_word_similarity but adding a weighting
        factor based in the computed Tf-idf when computing the synset similarities.
        """
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

    def wordnet_pairwise_word_similarity_weighted_all_possibilities(self, config, sentence1, sentence2):
        """
        Computes the Wordnet-based pairwise word similarity as wordnet_pairwise_word_similarity but instead of using
        only the most likely synset for each word in the sentence extracted with the Lesk algorithm, it computes the
        similarities using the 5 most likely synsets for each of the words (no Lesk used).
        Originally it used ALL synsets for a word, changed to 5 synsets for efficiency purposes.
        """
        lch = lambda w1, w2: w1.lch_similarity(w2)
        path = lambda w1, w2: w1.path_similarity(w2)
        wup = lambda w1, w2: w1.wup_similarity(w2)
        metrics = {'lch': lch, 'path': path, 'wup': wup}
        metric = config['metric'] if 'metric' in config else 'lch'

        def _compute_sentence_similarities(s1, s2):
            similarities = []
            for index1, word in enumerate(s1[2]):
                max_similarity = 0
                tag1 = s1[3][index1][0].lower()
                if tag1 == 'j': tag1 = 'a'
                if tag1 in ['v', 'n', 'r', 'a']:
                    synsets1 = wn.synsets(word, tag1)[:5]
                    for synset1 in synsets1:
                        for index2, other_word in enumerate(s2[2]):
                            tag2 = s2[3][index2][0].lower()
                            if tag2 == 'j': tag2 = 'a'
                            if tag2 in ['v', 'n', 'r', 'a']:
                                    synsets2 = wn.synsets(other_word, tag2)[:5]
                                    for synset2 in synsets2:
                                        if synset1.pos() == synset2.pos():
                                            similarity = metrics[metric](synset1, synset2)
                                            if similarity is not None:
                                                max_similarity = max(similarity, max_similarity)
                similarities.append(max_similarity*self.word_idfs[word])
            return similarities

        scores1 = _compute_sentence_similarities(sentence1, sentence2)
        scores2 = _compute_sentence_similarities(sentence2, sentence1)
        if scores1 == [] or scores2 == []:
            return 0
        score1, score2 = np.mean(scores1), np.mean(scores2)
        return np.mean([score1, score2])

    def number_overlap(self, config, sentence1, sentence2):
        """
        Computes the Jaccard distance between the sets of the cardinal numbers that appear in the two sentences.
        It tries to parse numbers expressed in words (for example 'two', 'three hundreds and forty-nine'), numbers using
        ',' (for example '16,432,970') and decimal numbers (for example '45,233.4123'). If the number can't be parsed,
        it's added as the corresponding string to the set.
        """
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

    def translation_similarity(self, config, sentence1, sentence2):
        """
        UNUSED METRIC
        Computes the Jaccard similarity of two sentences after having translated them to German or Spanish and
        to English again. It was going to use the result of the preprocessing step of translating the sentences, which
        in the end was dropped.
        """
        lang = config['language'] if 'language' in config else 'de'
        tokens1 = word_tokenize(sentence1[5 if lang == 'de' else 6])
        tokens2 = word_tokenize(sentence2[5 if lang == 'de' else 6])

        set1 = set(tokens1)
        set2 = set(tokens2)
        intersection = set1.intersection(set2)
        sizei = len(intersection)
        size1, size2 = len(set1), len(set2)
        try:
            return 2 * (1 / (size1 / sizei + size2 / sizei))
        except ZeroDivisionError:
            return 0

    def dependency_overlap(self, config, sentence1, sentence2):
        """
        Computes the Jaccard similarity between the sets of extracted dependencies between words in each sentence.
        It uses the sentence in the form (word1, relation, word2) where the words have been lemmatized.
        """
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
        """
        Computes the length of the longest common subsequence or substring of the two sentences, depending on the
        configuration.
        L. C. Subsequence example:
          ("We ate a delicious pizza", "We ate a not so delicious pizza") -> "We ate a delicious pizza"
        L. C. Substring example:
          ("We ate a delicious pizza", "We ate a not so delicious pizza") -> " delicious pizza"
        """
        mode = config['mode'] if 'mode' in config else 'subsequence'
        if mode == 'subsequence':
            return pylcs.lcs(sentence1[1], sentence2[1])
        else:  # mode == 'substring'
            return pylcs.lcs2(sentence1[1], sentence2[1])


