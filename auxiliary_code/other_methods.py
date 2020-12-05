
# Sentence data saved inside an array -> 0: original sentence; 1: sentence transformations; 2: tokens; 3: pos tags;  4: synsets


def pretty_print_sentence(sentence):
    labels = ['Original sentence', 'Sentence transformation', 'Tokens', 'Pos tags', 'Synsets']
    for index, data in enumerate(sentence):
        print(labels[index] + ':', data)

