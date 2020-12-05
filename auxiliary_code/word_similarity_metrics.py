
def run_word_similarity_metrics(metrics, *args):
    for metric in metrics:
        eval(metric['name'])(metric, *args)