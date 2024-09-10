# handle imports
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import mean_squared_error
import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression, ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

class TrainerClass:
    def __init__(self,
                 x_matrix: pd.DataFrame,
                 y_abundance_matrix: pd.DataFrame,
                 x_hold_out: pd.DataFrame,
                 y_hold_out: pd.DataFrame,
                 algorithm: str,

                 ):

        self.algorithm = algorithm
        self.X_input_matrix = x_matrix
        self.Y_target_abundance = y_abundance_matrix
        self.X_hold_out = x_hold_out
        self.Y_hold_out = y_hold_out
        self.model = None
        self.cv_results = None
        self.cv_mse_scores = None
        self.cv_rmse_scores = None
        self.predictions = None

        self.model_regression_dict = {

            'knn': KNeighborsRegressor(n_jobs=-1),
            'linear_regression': LinearRegression(),
            'random_forest': RandomForestRegressor(random_state=42, n_jobs=-1),
            'elastic_net': ElasticNet(random_state=42),
        }

        # define parameter grid for each model
        self.model_param_grid_dict = {
            'linear_regression': {},
            'knn': {
                'n_neighbors': [3, 5, 7, 10, 15, 20],
                'weights': ['uniform', 'distance'],
                'p': [1, 2]
            },
            'random_forest': {
                'estimator__max_depth': [5, 10, 15],
                'estimator__max_features': ['sqrt'],
                'estimator__min_samples_split': [2],
                'estimator__min_samples_leaf': [1]
            },
            'elastic_net': {
                'estimator__alpha': [0.5, 1, 2],
                'estimator__l1_ratio': [0.3, 0.5, 0.7]
            },
        }

    def initialize_model(self):
        """
        initialize model
        :return:
        """
        self.model = MultiOutputRegressor(self.model_regression_dict[self.algorithm], n_jobs=-1) if self.algorithm != "knn" else self.model_regression_dict[self.algorithm]
        print("model in use: ", self.model)


    def cross_validation(self):
        """
        Perform cross-validation with grid search and stratified splits.
        """
        # Initialize model
        self.initialize_model()

        # Create stratified folds
        _folds = 5

        # GridSearchCV with custom cv parameter
        grid_search = GridSearchCV(estimator=self.model,
                                   param_grid=self.model_param_grid_dict[self.algorithm],
                                   scoring='neg_mean_squared_error',
                                   cv=_folds,
                                   verbose=2,
                                   n_jobs=-1)

        grid_search.fit(self.X_input_matrix, self.Y_target_abundance)

        # Best parameters and scores
        print("Cross-Validation Done")
        self.model = grid_search.best_estimator_

        # Calculate and store MSE and RMSE scores
        self.cv_results = grid_search.cv_results_
        self.cv_mse_scores = -self.cv_results['mean_test_score']  # Scores are negative MSE
        self.cv_rmse_scores = np.sqrt(self.cv_mse_scores)


    def fit_predict_best_model(self):
        """
        Fit the model with the best parameters found during grid search on the entire dataset,
        then predict and evaluate on unseen (hold-out) data.

        Parameters:
        - X_hold_out: The features of the hold-out set.
        - Y_hold_out: The true target values of the hold-out set.

        Returns:
        - predictions: The predicted values for the hold-out set.
        - mse: Mean squared error for the hold-out set.
        - rmse: Root mean squared error for the hold-out set.

        """

        print("Best parameters found for final model: ", self.model.get_params())

        # Assuming self.model is already the best estimator from GridSearchCV
        # Fit the model on the entire dataset (self.X_input_matrix, self.Y_target_abundance)
        self.model.fit(self.X_input_matrix, self.Y_target_abundance)

        # Predict on the hold-out set
        self.prediction_matrix = self.model.predict(self.X_hold_out)
        self.predictions = self.prediction_matrix[0]
        self.predictions = pd.DataFrame(self.predictions, index=self.Y_hold_out.columns)

        # Calculate MSE and RMSE for the hold-out set
        self.best_model_mse = mean_squared_error(self.Y_hold_out, self.prediction_matrix, multioutput='raw_values')
        self.best_model_rmse = np.sqrt(self.best_model_mse)



    def run_train_predict_based_on_algorithm(self):
        """
        Run the entire training and prediction process based on the chosen algorithm.
        """
        self.cross_validation()
        self.fit_predict_best_model()

