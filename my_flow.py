"""

Simple stand-alone script showing end-to-end training of a regression model using Metaflow. 
This script ports the composable script into an explicit dependency graphg (using Metaflow syntax)
and highlights the advantages of doing so. This script has been created for pedagogical purposes, 
and it does NOT necessarely reflect all best practices.

Please refer to the slides and our discussion for further context.

MAKE SURE TO RUN THIS WITH METAFLOW LOCAL FIRST

"""


from metaflow import FlowSpec, step, Parameter, IncludeFile, current
from datetime import datetime
import os
import glob
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt

from joblib import Parallel, delayed

# make sure we are running locally for this
assert os.environ.get('METAFLOW_DEFAULT_DATASTORE', 'local') == 'local'
assert os.environ.get('METAFLOW_DEFAULT_ENVIRONMENT', 'local') == 'local'
from sklearn.neighbors import NearestNeighbors
def rmspe(y_true, y_pred):
    return np.sqrt(np.mean(np.square((y_true - y_pred) / y_true)))
def feval_rmspe(preds, train_data):
    labels = train_data.get_label()
    return 'RMSPE', round(rmspe(y_true = labels, y_pred = preds),5), False
class Neighbors:
    def __init__(self, name: str, n_neighbor: int, pivot: pd.DataFrame, X_train: pd.DataFrame):
        self.features = pd.DataFrame()
        self.name = name
        self.n_neighbor = n_neighbor
        self.pivot = pivot
        nn = NearestNeighbors(n_neighbors=self.n_neighbor)
        nn.fit(pivot)
        self.nn_ind = nn.kneighbors(pivot, n_neighbors=self.n_neighbor, return_distance=False)
        self.time = X_train.time_id.factorize()[0]
        self.stock = X_train.stock_id.factorize()[0]  
    
    def make_nn_features(self, n: int, col: str, features: pd.DataFrame, method=np.mean):
        assert(self.features is not None)

        self.agg = pd.DataFrame(
            method(self.features.iloc[:, 1:n], axis = 1),
            columns = [f'{self.name}_{str(n)}_{col}_{method.__name__}']
        )
    
class TimeNeighbors(Neighbors):
    def make_nn_time(self, pivot: pd.DataFrame, feature_col: str):
        pivot = pivot.pivot('time_id', 'stock_id', feature_col)
        pivot = pivot.fillna(pivot.mean())
        self.features = pd.DataFrame(pivot.values[self.nn_ind[self.time, 1:self.n_neighbor], self.stock[:, None]])
        self.col = feature_col

class StockNeighbors(Neighbors):        
    def make_nn_stock(self, pivot: pd.DataFrame, feature_col: str):
        pivot = pivot.pivot('time_id', 'stock_id', feature_col)
        pivot = pivot.fillna(pivot.mean())
        self.features = pd.DataFrame(pivot.T.values[self.nn_ind[self.stock, 1:self.n_neighbor], self.time[:, None]])
        self.col = feature_col

class MyFlow(FlowSpec):
    """
    MyRegressionFlow is a minimal DAG showcasing reading data from a file 
    and training a model successfully.
    """
    
    # if a static file is part of the flow, 
    # it can be called in any downstream process,
    # gets versioned etc.
    # https://docs.metaflow.org/metaflow/data#data-in-local-files
    """
    DATA_FILE = IncludeFile(
        'dataset',
        help='csv file with the dataset',
        is_text=True,
        default='df.csv')
    TRAIN_FILE = IncludeFile(
        'dataset_train',
        help='csv file with the dataset',
        is_text=True,
        default='./data/train.csv')
   """

    @step
    def start(self):
        """
        Start up and print out some info to make sure everything is ok metaflow-side
        """
        print("Starting up at {}".format(datetime.utcnow()))
        # debug printing - this is from https://docs.metaflow.org/metaflow/tagging
        # to show how information about the current run can be accessed programmatically
        print("flow name: %s" % current.flow_name)
        print("run id: %s" % current.run_id)
        print("username: %s" % current.username)
        self.next(self.load_data)

    @step
    def load_data(self): 
        """
        Read the data in from the static file
        """
        from io import StringIO
        # load train file
        self.train = pd.read_csv('./data/train.csv')
        print(self.train.shape)
        self.stock_ids = list(set(self.train['stock_id']))
        self.stock_ids = self.stock_ids[:10]
        print(self.stock_ids)
        # load preprocessed data file (~30minutes of preprocessing)
        self.df = pd.read_csv('./df_v1.csv')
        print(self.df.shape)
        self.time_ids = self.df.time_id.factorize()[1]
        time_len = len(self.time_ids)
        html_time = 0.1
        train_time = self.time_ids[int((1 - html_time) * time_len)]
        self.X_train = self.df.copy()
        self.X_train['target'] = self.train.target
        print(self.X_train['target'])
        # go to the next step
        self.next(self.make_nn_features)
    @step
    def make_nn_features(self):
        from sklearn.preprocessing import minmax_scale
        time_neighbor = []
        stock_neighbor = []
        pv = self.X_train.copy()
        pivot = pv.pivot('time_id','stock_id','book_log_return1_realized_volatility')
        pivot = pivot.fillna(pivot.mean())
        pivot = pd.DataFrame(minmax_scale(pivot))
        time_neighbor.append(TimeNeighbors('time_vol', 4, pivot, self.X_train))

        pivot = pv.pivot('time_id','stock_id','trade_size_sum')
        pivot = pivot.fillna(pivot.mean())
        pivot = pd.DataFrame(minmax_scale(pivot))
        time_neighbor.append(TimeNeighbors('time_trade_size_sum', 4, pivot, self.X_train))

        pivot = pv.pivot('time_id','stock_id','book_log_return1_mean')
        pivot = pivot.fillna(pivot.mean())
        pivot = pd.DataFrame(minmax_scale(pivot))
        time_neighbor.append(TimeNeighbors('time_log_return1_mean', 4, pivot, self.X_train))

        pivot = pv.pivot('time_id','stock_id','trade_order_count_mean')
        pivot = pivot.fillna(pivot.mean())
        pivot = pd.DataFrame(minmax_scale(pivot))
        time_neighbor.append(TimeNeighbors('time_order_count_mean', 4, pivot, self.X_train))

        pivot = pv.pivot('time_id','stock_id','book_log_return1_realized_volatility')
        pivot = pivot.fillna(pivot.mean())
        pivot = pd.DataFrame(minmax_scale(pivot))
        stock_neighbor.append(StockNeighbors('stock_vol', 8, minmax_scale(pivot.T), self.X_train))

        pivot = pv.pivot('time_id','stock_id','book_log_return1_mean')
        pivot = pivot.fillna(pivot.mean())
        pivot = pd.DataFrame(minmax_scale(pivot))
        stock_neighbor.append(StockNeighbors('stock_log_return1_mean', 8, minmax_scale(pivot.T), self.X_train))

        self.df_nn = self.X_train.copy()
        #print("df")
        #print(self.df['target'])
        methods_stocks = {
            'book_log_return1_realized_volatility': [np.mean, np.min, np.max, np.std],
            'trade_order_count_mean': [np.mean],
            'book_log_return1_mean': [np.mean],
            'trade_size_sum': [np.mean],
            'book_seconds_in_bucket_count': [np.mean],
        }
    
        methods_time = {
            'book_log_return1_realized_volatility': [np.mean, np.min, np.max, np.std],
            'trade_order_count_mean': [np.mean],
            'book_log_return1_mean': [np.mean],
            'trade_size_sum': [np.mean],
            'book_seconds_in_bucket_count': [np.mean],
        }
        time_n = [2, 3, 5, 10, 20, 40]
        stock_n = [2,4,6,8,9]
        cols = []
        for col in methods_time.keys():

            for nn in time_neighbor:
                nn.make_nn_time(self.df_nn, col)

            for method in methods_time[col]:
                for n in time_n:
                    nn.make_nn_features(n, col, nn.features, method)
                    feat = nn.agg
                    cols.append(feat)
        for col in methods_stocks.keys():
            for nn in stock_neighbor:
                nn.make_nn_stock(self.df_nn, col)

            for method in methods_stocks[col]:
                for n in stock_n:
                    nn.make_nn_features(n, col, nn.features, method)
                    feat = nn.agg
                    cols.append(feat)    
    
        ndf = pd.concat(cols, axis = 1)
        self.df_nn = pd.concat([self.df_nn, ndf], axis = 1)
        for i in range(len(self.df_nn.columns)):
            if self.df_nn.columns[i] == 'target':
                print("target!!!!!!!!!!!!!!!!!!!!!")
        self.df_nn = pd.read_csv('./df_nn.csv')
        self.df_nn = self.df_nn.fillna(self.df_nn.mean())
        self.next(self.prepare_train_and_test_dataset)

    @step
    def prepare_train_and_test_dataset(self):
        from sklearn.model_selection import train_test_split

        fold_bolder = [3830 - 383 * 5, 3830 - 383 * 4,3830 - 383 * 3,3830 - 383 * 2,3830 - 383 * 1,]
        self.fold = []
        self.df_nn = self.df_nn.sort_values(by = ['time_id', 'stock_id'])
        self.df_nn = self.df_nn.reset_index()
        for i in fold_bolder:
            idx_valid = self.time_ids[i: i + 383]
            ind_train = np.where(self.df_nn.time_id<idx_valid[0])[0]
            ind_valid = np.where((self.df_nn.time_id>=idx_valid[0])&(self.df_nn.time_id<idx_valid[-1]))[0]
            self.fold.append((ind_train, ind_valid))
        self.df_train = self.df_nn[self.df_nn.index <= self.fold[-1][0][-1]]
        self.df_test = self.df_nn[self.df_nn.index > self.fold[-1][0][-1]]
        self.df_train = pd.read_csv('./df_train.csv')
        self.df_test = pd.read_csv('./df_test.csv')
        print(self.df_train.shape)
        self.X = self.df_train[[col for col in self.df_train.columns if col not in ['target','time_id','stock_id','level_0','index']]]
        self.y = self.df_train['target']
        self.X_test = self.df_test[[col for col in self.df_test.columns if col not in ['target','time_id','stock_id','level_0','index']]]
        self.y_test = self.df_test['target']
        print(self.y.head())
        self.next(self.set_params)
    @step
    def set_params(self):
        self.params = [{
            'objective': 'regression',
            'verbose': 0,
            'metric': '',
            'reg_alpha': 5,
            'reg_lambda': 5,
            'min_data_in_leaf': 1000,
            'max_depth': -1,
            'num_leaves': 128,
            'colsample_bytree': 0.3,
            'learning_rate': 0.3
            },
            {
            'objective': 'regression',
            'verbose': -1,
            'metric': 'None',
            'reg_alpha': 5,
            'reg_lambda': 5,
            'min_data_in_leaf': 256,
            'max_depth': 8,
            'num_leaves': 800,
            'colsample_bytree': 0.3,
            'learning_rate': 0.01
            },
            {
            'objective': 'regression',
            'verbose': 0,
            'metric': '',
            'reg_alpha': 5,
            'reg_lambda': 5,
            'min_data_in_leaf': 1000,
            'max_depth': 12,
            'num_leaves': 300,
            'colsample_bytree': 0.5,
            'learning_rate': 0.2
            },
            {
            'objective': 'regression',
            'verbose': -1,
            'metric': '',
            'reg_alpha': 5,
            'reg_lambda': 5,
            'min_data_in_leaf': 128,
            'max_depth': 12,
            'num_leaves': 1000,
            'colsample_bytree': 0.2,
            'learning_rate': 0.1
            },
        ]
        self.next(self.train_model, foreach='params')
    @step
    def train_model(self):
        """
        Train a regression on the training set
        """
        import lightgbm as lgb
        
        self.param = self.input
        print(self.param)
        self.ds = lgb.Dataset(self.X, self.y, weight = 1/np.power(self.y, 2))
        
        from sklearn.model_selection import KFold
        self.models = []
        self.model = None
        self.best_RMSPE = 2**31
        x_train, x_val = self.X.iloc[self.fold[3][0]], self.X.iloc[self.fold[3][1]]
        y_train, y_val = self.y.iloc[self.fold[3][0]], self.y.iloc[self.fold[3][1]]
        # Root mean squared percentage error weights
        train_weights = 1 / np.square(y_train)
        val_weights = 1 / np.square(y_val)
        train_dataset = lgb.Dataset(x_train, y_train, weight = train_weights)
        val_dataset = lgb.Dataset(x_val, y_val, weight = val_weights)
        model = lgb.train(params = self.param,
                          num_boost_round=1000,
                          train_set = train_dataset, 
                          valid_sets = [train_dataset, val_dataset], 
                          verbose_eval = 250,
                          early_stopping_rounds=500,
                          feval = feval_rmspe)
        # Add predictions to the out of folds array
        ypred = model.predict(x_val)
        rmspe_score = rmspe(y_val, ypred)
        print(f'Our out of folds RMSPE is {rmspe_score}')
        self.train_val_result = {"RMSPE": rmspe_score, "model": model,"param":self.param, "X_test": self.X_test, "y_test": self.y_test}
        self.next(self.join)

    @step
    def join(self, inputs):
        self.train_val_results = [input.train_val_result for input in inputs]
        self.best_model = min(self.train_val_results, key=lambda train_val_result: train_val_result["RMSPE"])
        self.best_param = self.best_model["param"]
        self.best_lgb_model = self.best_model["model"]
        self.X_test = self.best_model['X_test']
        self.y_test = self.best_model['y_test']
        self.next(self.test_model)
    @step 
    def test_model(self):
        """
        Test the model on the hold out sample
        """
        self.y_pred = self.best_lgb_model.predict(self.X_test, num_iteration=self.best_lgb_model.best_iteration)
        print(f"# RMSPE: {np.sqrt(np.mean(np.square((self.y_test - self.y_pred) / self.y_test)))}")                    
        self.next(self.end)

    @step
    def end(self):
        # all done, just print goodbye
        print("All done at {}!\n See you, space cowboys!".format(datetime.utcnow()))


if __name__ == '__main__':
    print("enter flow")
    MyFlow()
