import re


def load_train_data():
    train_values = []
    train_labels = []
    train_path = './input/train'

    train_values = load_values_file(train_path + '/STS.input.MSRpar.txt', train_values)
    train_values = load_values_file(train_path + '/STS.input.MSRvid.txt', train_values)
    train_values = load_values_file(train_path + '/STS.input.SMTeuroparl.txt', train_values)

    train_labels = load_labels_file(train_path + '/STS.gs.MSRpar.txt', train_labels)
    train_labels = load_labels_file(train_path + '/STS.gs.MSRvid.txt', train_labels)
    train_labels = load_labels_file(train_path + '/STS.gs.SMTeuroparl.txt', train_labels)

    return train_values, train_labels


def load_test_data():
    test_values = []
    test_labels = []
    test_path = './input/test-gold'

    test_values = load_values_file(test_path + '/STS.input.MSRpar.txt', test_values)
    test_values = load_values_file(test_path + '/STS.input.MSRvid.txt', test_values)
    test_values = load_values_file(test_path + '/STS.input.SMTeuroparl.txt', test_values)
    test_values = load_values_file(test_path + '/STS.input.surprise.OnWN.txt', test_values)
    test_values = load_values_file(test_path + '/STS.input.surprise.SMTnews.txt', test_values)

    test_labels = load_labels_file(test_path + '/STS.gs.MSRpar.txt', test_labels)
    test_labels = load_labels_file(test_path + '/STS.gs.MSRvid.txt', test_labels)
    test_labels = load_labels_file(test_path + '/STS.gs.SMTeuroparl.txt', test_labels)
    test_labels = load_labels_file(test_path + '/STS.gs.surprise.OnWN.txt', test_labels)
    test_labels = load_labels_file(test_path + '/STS.gs.surprise.SMTnews.txt', test_labels)

    return test_values, test_labels


def load_values_file(path, array):
    with open(path, encoding='utf8') as f:
        data = f.read()
        splitted_data = re.split('\n+', data) # split raw data by tabs and end of lines
        if splitted_data[-1] == '':
            del splitted_data[-1] # delete last list empty value caused by splitting the last end of line
        for pair in splitted_data:
            sentence1, sentence2 = pair.split('\t')
            array.append((sentence1, sentence2))
    return array


def load_labels_file(path, array):
    with open(path, encoding='utf8') as f:
        data = f.read()
        splitted_data = re.split('\n+', data) # split raw data by tabs and end of lines
        if splitted_data[-1] == '':
            del splitted_data[-1] # delete last list empty value caused by splitting the last end of line
        for score in splitted_data:
            array.append(float(score))
    return array