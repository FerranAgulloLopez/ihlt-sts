from scipy.stats import pearsonr
import numpy as np
import argparse
import logging
from time import time

from auxiliary_code.file_methods import load_train_data, load_test_data, load_json, save_json
from auxiliary_code.preprocessing_steps import Preprocessing
from auxiliary_code.other_methods import pretty_print_sentence
from auxiliary_code.sentence_similarity_metrics import SentenceSimilarity
from auxiliary_code.visualize import show_scatter_plot, show_correlation_plot
from auxiliary_code.aggregation_methods import AggregationMethodFactory


def parse_arguments():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--config_path', help="Path to config file", required=True)
    parser.add_argument('--output_path', default=None, help="Path to output directory", required=False)
    parser.add_argument('--type', default='full', help="Define if only run the similarity metrics, the aggregation method or both of them", required=False)
    parser.add_argument('--input_path', default=None, help="Path to input directory when using aggregation as type", required=False)
    return parser.parse_args()


def create_logger(file_path):
    # Create logger to print in stdout and in file if required
    logger = logging.getLogger()
    if file_path is not None:
        file_handler = logging.FileHandler(file_path + '/log.txt')
        logger.addHandler(file_handler)
    console_handler = logging.StreamHandler()
    logger.addHandler(console_handler)
    logger.setLevel(logging.INFO)
    return logger


def run_similarity_metrics(config, train_sentence_pairs, train_labels, test_sentence_pairs, test_labels, logger, output_path):

    # Preprocess data
    preprocessing = Preprocessing(config['preprocessing_steps'])
    train_output = preprocessing.do_pipeline(train_sentence_pairs)
    test_output = preprocessing.do_pipeline(test_sentence_pairs)

    # Show example of two sentences
    logger.info('\nSentence 1')
    pretty_print_sentence(train_output[0][0], logger)
    logger.info('\nSentence 2')
    pretty_print_sentence(train_output[0][1], logger)

    # Compute metric values
    sentence_similarity = SentenceSimilarity(config['similarity_metrics'])
    metric_train_values = sentence_similarity.compute_pair_comparison(train_output)
    metric_test_values = sentence_similarity.compute_pair_comparison(test_output)

    # Show correlation between metrics including the gold standard
    show_correlation_plot(config['similarity_metrics'], train_labels, metric_train_values,
                          'Similiarity metrics correlation to the goldan standard in the train set', output_path)
    show_correlation_plot(config['similarity_metrics'], test_labels, metric_test_values,
                          'Similiarity metrics correlation to the goldan standard in the test set', output_path)

    return metric_train_values, metric_test_values


def run_aggregation_method(config, metric_train_values, train_labels, metric_test_values, test_labels, logger, output_path):

    aggregation_method = AggregationMethodFactory.select_aggregation_method(config, metric_train_values, train_labels, metric_test_values, logger)
    train_results = aggregation_method.train()
    test_results = aggregation_method.test()

    logger.info('\nCorrelation with Gold standard in the training set')
    logger.info(pearsonr(train_labels, train_results)[0])
    logger.info('\nCorrelation with Gold standard in the testing set')
    logger.info(pearsonr(test_labels, test_results)[0])

    show_scatter_plot(train_labels, train_results, 'Similarity vs Gold standard in the training set', output_path)
    show_scatter_plot(test_labels, test_results, 'Similarity vs Gold standard in the testing set', output_path)

    return train_results, test_results


def main(config_path, output_path, _type, input_path):
    logger = create_logger(output_path)
    config = load_json(config_path)

    if _type == 'full' or _type == 'metrics':

        # Load data
        train_sentence_pairs, train_labels = load_train_data()
        test_sentence_pairs, test_labels = load_test_data()
        logger.info('Train values length: ' + str(len(train_sentence_pairs)) + '; Train labels length: ' + str(len(train_labels)))
        logger.info('Test values length: ' + str(len(test_sentence_pairs)) + '; Test labels length: ' + str(len(test_labels)))

        # Run metrics
        ini_time = time()
        metric_train_values, metric_test_values = run_similarity_metrics(config, train_sentence_pairs, train_labels, test_sentence_pairs, test_labels, logger, output_path)
        logger.info('\nElapsed time during metrics computation: ' + str(time() - ini_time))
        if output_path is not None:
            np.save(output_path + '/metric_train_values', metric_train_values)
            np.save(output_path + '/metric_test_values', metric_test_values)
            np.save(output_path + '/train_labels', train_labels)
            np.save(output_path + '/test_labels', test_labels)
            save_json(output_path + '/config', config)

        if _type == 'full':
            # Run aggregation method
            ini_time = time()
            train_results, test_results = run_aggregation_method(config, metric_train_values, train_labels, metric_test_values, test_labels, logger, output_path)
            logger.info('\nElapsed time during aggregation computation: ' + str(time() - ini_time))
            if output_path is not None:
                np.save(output_path + '/train_results',train_results)
                np.save(output_path + '/test_results', test_results)

    elif _type == 'aggregation':

        # Load data
        if input_path is None: raise Exception('Type aggregation requires an output path to load the input data from')
        metric_train_values = np.load(input_path + '/metric_train_values.npy')
        metric_test_values = np.load(input_path + '/metric_test_values.npy')
        train_labels = np.load(input_path + '/train_labels.npy')
        test_labels = np.load(input_path + '/test_labels.npy')

        # Run aggregation method
        ini_time = time()
        train_results, test_results = run_aggregation_method(config, metric_train_values, train_labels, metric_test_values, test_labels, logger, output_path)
        logger.info('\nElapsed time during aggregation computation: ' + str(time() - ini_time))
        if output_path is not None:
            np.save(output_path + '/train_results', train_results)
            np.save(output_path + '/test_results', test_results)
            np.save(output_path + '/metric_train_values', metric_train_values)
            np.save(output_path + '/metric_test_values', metric_test_values)
            np.save(output_path + '/train_labels', train_labels)
            np.save(output_path + '/test_labels', test_labels)
            save_json(output_path + '/config', config)
    else:
        raise Exception('Type not recognized, read README file for details')


if __name__ == '__main__':
    args = parse_arguments()
    main(args.config_path, args.output_path, args.type, args.input_path)



