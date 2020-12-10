# IHLT - Project - Semantic Textual Similarity

_This component was created as a result of the IHLT subject in the MAI Masters degree._

## Introduction

This README is divided into the following 4 sections:
- Funcionality: We explain the functionality of the code
- Files structure: We explain the files and directories purposes that are included in this code
- How to install: We explain how to set up the code
- How to run: We explain how to use the code

### Funcionality

The funcionality of the component is to run experiments with different similarity metrics to obtain a good score in the STS 2012 evaluation.

Our approach contains the next three different steps:

- Preprocessing: in this step we run different methods like tokenization and punctuation removal to prepare the data for the following steps.
- Similarity metrics computation: in this step we compute a set of similarity metrics between the sentences like word n-gram jaccard similarity and wordnet pairwise word similarity.
- Aggregation method: in this step we group the prior similarity metrics with a method like random forest or a support vector machine to obtain a final score to compute the correlation with the gold standard.

We use a Json object to define all the parameters of each experiment. In this way, it is only necessary to modify the json object to change the experiment configuration without touching any code. There are examples of configuration Jsons in the directory with path ./input/config_examples.

### Files structure

- input: directory containing configuration examples and the data from the STS 2012 evaluation
- output: directory containing the results of the experiments included in the presentation
- notebook: notebook containing the code for running the experiment with which we have obtained the best correlation score
- run_script: python script to run experiments in a much easier fashion than the notebook
- auxiliary_code: directory containing the important code, the implementation of all the preprocessing steps, similarity metrics and aggregation methods
    - preprocessing_steps: python file containing the code that implements the multiple preprocessing steps
    - sentence_similarity_metrics: python file containing the code that implements the multiple similarity metrics
    - aggregation_methods: python file containing the code that implements the multiple aggregation methods
    - visualize: python file containing the code that implements the creation of some charts
    - file_methods: python file containing the code that implements the loading of the data files
    - other_methods: python file containing the code that implements some useful methods

### How to install

- Use an environment with python3.6
- Install the libraries in the requirements.txt file
- Run a StanfordCoreNLPServer in the port 9000 if you want to use the dependency_overlap similarity metric

### How to run

#### Notebook

It is not necessary any special requirements to run the notebook

#### Script file

It is necessary to run the run_script.py file with the following parameters:
- config_path: json file with all the parameters that define the experiment to run
- output_path: (optional) path defining the directory to save the experiment results

An example:
- python3 run_script.py --config_path ./input/config_examples/test.json
