import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler, PowerTransformer
from sklearn.svm import SVR
from sklearn.kernel_ridge import KernelRidge
from sklearn.ensemble import AdaBoostRegressor, RandomForestRegressor, ExtraTreesRegressor, VotingRegressor
from sklearn.decomposition import PCA


class AggregationMethod():

    def __init__(self, config, train_values, train_labels, test_values, logger):
        # Standarize values
        mms_values = MinMaxScaler()
        mms_values.partial_fit(train_values)
        mms_values.partial_fit(test_values)
        self.train_values = mms_values.transform(train_values)
        self.test_values = mms_values.transform(test_values)
        self.train_labels = train_labels
        self.logger = logger

        if 'reduction' in config['aggregation_method'] and config['aggregation_method']['reduction']['name'] == 'pca':
            self.train_values, self.test_values = self.do_pca_reduction(config['aggregation_method']['reduction'], self.train_values, self.test_values)

    def train(self):
        return self.standarize_predictions(self.train_model(self.train_values, self.train_labels))[:,0]

    def test(self):
        return self.standarize_predictions(self.test_model(self.test_values))[:,0]

    def standarize_predictions(self, predictions):
        mms_pred = MinMaxScaler(feature_range=(0, 5))
        return mms_pred.fit_transform(np.reshape(predictions, (len(predictions), 1)))

    def do_pca_reduction(self, config, train_values, test_values):
        pca = PCA(n_components=config['n_components'])
        pca.fit(train_values)
        return pca.transform(train_values), pca.transform(test_values)

    def train_model(self, values, labels):
        raise Exception('Method not implemented in abstract class')

    def test_model(self, values):
        raise Exception('Method not implemented in abstract class')


class MeanAggregationMethod(AggregationMethod):

    def train_model(self, values, labels):
        return values.mean(axis=1)

    def test_model(self, values):
        return values.mean(axis=1)


class MaxAggregationMethod(AggregationMethod):

    def train_model(self, values, labels):
        return values.max(axis=1)

    def test_model(self, values):
        return values.max(axis=1)


class SVMAggregationMethod(AggregationMethod):

    def __init__(self, config, train_values, train_labels, test_values, logger):
        super().__init__(config, train_values, train_labels, test_values, logger)
        self.model = GridSearchCV(SVR(kernel='rbf', gamma=0.1), param_grid={"C": [0.1, 1, 10, 100, 1000],
                                                                "gamma": np.logspace(-2, 2, 5)})

    def train_model(self, values, labels):
        self.model = self.model.fit(values, labels)
        self.logger.info('\nAggregation method best params: ' + str(self.model.best_params_))
        return self.model.predict(values)

    def test_model(self, values):
        return self.model.predict(values)


class KRRAggregationMethod(AggregationMethod):

    def __init__(self, config, train_values, train_labels, test_values, logger):
        super().__init__(config, train_values, train_labels, test_values, logger)
        self.model = GridSearchCV(KernelRidge(kernel='rbf', gamma=0.1), param_grid={"alpha": [10, 1e0, 0.1, 1e-2, 1e-3],
                                                                                    "gamma": np.logspace(-2, 2, 5)})

    def train_model(self, values, labels):
        self.model = self.model.fit(values, labels)
        self.logger.info('\nAggregation method best params: ' + str(self.model.best_params_))
        return self.model.predict(values)

    def test_model(self, values):
        return self.model.predict(values)


class AdaBoostAggregationMethod(AggregationMethod):

    def __init__(self, config, train_values, train_labels, test_values, logger):
        super().__init__(config, train_values, train_labels, test_values, logger)
        self.model = GridSearchCV(AdaBoostRegressor(), param_grid={"n_estimators": [25, 50, 75, 100],
                                                                "learning_rate": [0.01, 0.1, 1]})

    def train_model(self, values, labels):
        self.model = self.model.fit(values, labels)
        self.logger.info('\nAggregation method best params: ' + str(self.model.best_params_))
        return self.model.predict(values)

    def test_model(self, values):
        return self.model.predict(values)


class RandomForestAggregationMethod(AggregationMethod):

    def __init__(self, config, train_values, train_labels, test_values, logger):
        super().__init__(config, train_values, train_labels, test_values, logger)
        self.model = GridSearchCV(RandomForestRegressor(), param_grid={"n_estimators": [100],
                                                                #"criterion": ["mse", "mae"],
                                                                "criterion": ["mse"],
                                                                #"min_samples_leaf": [1, 0.1, 0.2, 0.4, 0.5],
                                                                #"min_samples_split": [2, 0.2, 0.4, 0.6, 0.8],
                                                                "min_samples_split": [2, 0.2, 0.5, 0.8],
                                                                #"max_features": ["auto", "sqrt", "log2"]
                                                                "max_features": ["auto"]})

    def train_model(self, values, labels):
        self.model = self.model.fit(values, labels)
        self.logger.info('\nAggregation method best params: ' + str(self.model.best_params_))
        return self.model.predict(values)

    def test_model(self, values):
        return self.model.predict(values)


class VotingAggregationMethod(AggregationMethod):

    def __init__(self, config, train_values, train_labels, test_values, logger):
        super().__init__(config, train_values, train_labels, test_values, logger)
        self.model = VotingRegressor([
            ('random_forest', SVR(kernel='rbf', gamma=0.1)),
            ('krr', KernelRidge(kernel='rbf', gamma=0.1)),
            ('ada', AdaBoostRegressor()),
            ('rf', RandomForestRegressor()),
            ('et', ExtraTreesRegressor())
        ])

    def train_model(self, values, labels):
        self.model = self.model.fit(values, labels)
        return self.model.predict(values)

    def test_model(self, values):
        return self.model.predict(values)


class AggregationMethodFactory():

    def __init__(self):
        raise Exception('This class can not be instantiated')

    @staticmethod
    def select_aggregation_method(config, *args) -> AggregationMethod:
        name = config['aggregation_method']['name']
        if name == 'mean':
            aggregation_method = MeanAggregationMethod(config, *args)
        elif name == 'max':
            aggregation_method = MaxAggregationMethod(config, *args)
        elif name == 'svm':
            aggregation_method = SVMAggregationMethod(config, *args)
        elif name == 'krr':
            aggregation_method = KRRAggregationMethod(config, *args)
        elif name == 'ada_boost':
            aggregation_method = AdaBoostAggregationMethod(config, *args)
        elif name == 'random_forest':
            aggregation_method = RandomForestAggregationMethod(config, *args)
        elif name == 'voting':
            aggregation_method = VotingAggregationMethod(config, *args)
        else:
            raise Exception('The aggregation_method with name ' + name + ' does not exist')
        return aggregation_method




