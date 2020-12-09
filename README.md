# ihlt-sts

explain how to run -> run corenlp server +++

run full pipeline:
--config_path ./input/config_examples/test.json --output_path ./output

only run metrics:
--config_path ./input/config_examples/test.json --output_path ./output --type metrics

only run aggregation:
--config_path ./input/config_examples/test.json --output_path ./output --input_path ./output/test_metrics --type aggregation
