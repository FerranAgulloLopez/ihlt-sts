import nltk
import string

# Sentence data saved inside an array -> 0: original sentence; 1: sentence transformations; 2: tokens; 3: pos tags;  4: synsets


class Preprocessing:

    def __init__(self, config):
        self.steps = config
        step_names = {x['name'] for x in config}

        # Load the more common punctuation symbols
        if 'punctuation_removal' in step_names:
            self.punctuation_set = set(string.punctuation)
            self.punctuation_set.add('``')
            self.punctuation_set.add('\'\'')

        # Load english stop words
        if 'stopwords_removal' in step_names:
            self.sw_set = set(nltk.corpus.stopwords.words('english'))

        # Create lemmatizer
        if 'lemmatization' in step_names:
            self.wnl = nltk.stem.WordNetLemmatizer()

    # Main methods

    def do_pipeline(self, sentence_pairs):
        """
        Preprocesses all sentence pairs in sentence_pairs.
        """
        output = []
        for pair in sentence_pairs:
            sentence1, sentence2 = pair
            sentence1 = [sentence1, sentence1]
            sentence2 = [sentence2, sentence2]
            data1 = self.run_preprocessing_steps(self.steps, sentence1)
            data2 = self.run_preprocessing_steps(self.steps, sentence2)
            output.append((data1, data2))
        return output

    def run_preprocessing_steps(self, steps, sentence):
        """
        Preprocesses a pair of sentences, using the steps in the steps parameter.
        """
        for step in steps:
            sentence = self.run_preprocessing_step(step, sentence)
        return sentence

    def run_preprocessing_step(self, step, sentence):
        """
        Performs one preprocessing step to a sentence calling the corresponding function.
        """
        return eval('self.' + step['name'])(sentence)

    # Auxiliary methods

    def lower_case(self, sentence):
        """
        Lowercases a sentence.
        """
        # pre: the sentence in one string
        sentence[1] = sentence[1].lower()
        return sentence

    def word_tokenize(self, sentence):
        """
        Tokenizes a sentence.
        """
        # pre: the sentence in one string
        sentence.append(nltk.word_tokenize(sentence[1]))
        return sentence

    def punctuation_removal(self, sentence):
        """
        Removes the punctuation of a sentence, using the token list.
        """
        output = []
        for token in sentence[2]:
            if token not in self.punctuation_set:
                output.append(token)
        sentence[2] = output
        return sentence

    def stopwords_removal(self, sentence):
        """
        Removes stopwords in a sentence, using the token list.
        """
        output = []
        for token in sentence[2]:
            if token not in self.sw_set:
                output.append(token)
        sentence[2] = output
        return sentence

    def pos_tagging(self, sentence):
        """
        Tags the part of speech of each word in a sentence.
        """
        pairs = nltk.pos_tag(sentence[2])
        tags = [tag for token, tag in pairs]
        sentence.append(tags)
        return sentence

    def lemmatization(self, sentence):
        """
        Lemmatizes a sentence, using the tokens and PoS.
        """
        output = []
        for index, token in enumerate(sentence[2]):
            tag = sentence[3][index]
            try:
                output.append(self.wnl.lemmatize(token, pos=tag[0].lower()))
            except:
                output.append(token)
        sentence[2] = output
        return sentence

    def word_sense_disambiguation(self, sentence):
        """
        Performs word sense disambiguation using the Lesk algorithm to get a list of synsets. If a word doesn't have
        a synset, the original word is preserved.
        """
        tokens = sentence[2]
        tags = sentence[3]
        # TODO update
        output = []
        for index, token in enumerate(tokens):
            tag = tags[index]
            synset = nltk.wsd.lesk(tokens, token, tag[0].lower())
            if synset is None:
                output.append(token)
            else:
                output.append(synset)
        sentence.append(output)
        return sentence
