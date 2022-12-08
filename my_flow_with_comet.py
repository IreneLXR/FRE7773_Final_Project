"""

Simple stand-alone script showing end-to-end training of a machine learning model using Metaflow. 
This script ports the composable script into an explicit dependency graphg (using Metaflow syntax)
and highlights the advantages of doing so. 

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
from comet_ml import Experiment
from comet_ml import init
import comet_ml
from sklearn.neighbors import NearestNeighbors

# MAKE SURE THESE VARIABLES HAVE BEEN SET
os.environ['COMET_API_KEY']="BiFnq5zTOzwmQb2ZNMsJdAVUP"
os.environ['MY_PROJECT_NAME'] = "finalproject7773-yu-gu"
assert 'COMET_API_KEY' in os.environ and os.environ['COMET_API_KEY']
assert 'MY_PROJECT_NAME' in os.environ and os.environ['MY_PROJECT_NAME']
print("Running experiment for project: {}".format(os.environ['MY_PROJECT_NAME']))

# Create an experiment with your api key
experiment = Experiment(
    api_key="BiFnq5zTOzwmQb2ZNMsJdAVUP",
    project_name="finalproject7773-yu-gu",
    workspace="nyu-fre-7773-2021",
)


def rmspe(y_true, y_pred):
    return np.sqrt(np.mean(np.square((y_true - y_pred) / y_true)))
def feval_rmspe(preds, train_data):
    labels = train_data.get_label()
    return 'RMSPE', round(rmspe(y_true = labels, y_pred = preds),5), False

# Train NN, make NN features
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
    MyFlow is used to utilize LightGBM to predict future volatility.
    """
    """
    DATA_FILE = IncludeFile(
        'dataset',
        help='csv file with the dataset',
        is_text=True,
        default='./df.csv')
   
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
        self.stock_ids = list(set(self.train['stock_id']))
        self.stock_ids = self.stock_ids[:10]
        # load preprocessed data file (~30minutes of preprocessing)
        self.df = pd.read_csv('./data/df.csv', index_col=[0])
        self.time_ids = self.df.time_id.factorize()[1]
        self.X_train = self.df.copy()
        self.X_train['target'] = self.train.target
        import gc
        gc.collect()
        del self.train
        # go to the next step
        self.next(self.make_nn_features)
    @step
    def make_nn_features(self):
        """
        Process data, making nn features
        """
        from sklearn.preprocessing import minmax_scale
        time_neighbor = []
        stock_neighbor = []
        pv = self.X_train.copy()
        # make time neighbors based on volatility, trading volume and return
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
        
        # make stock neighbors based on volatility and return
        pivot = pv.pivot('time_id','stock_id','book_log_return1_realized_volatility')
        pivot = pivot.fillna(pivot.mean())
        pivot = pd.DataFrame(minmax_scale(pivot))
        stock_neighbor.append(StockNeighbors('stock_vol', 8, minmax_scale(pivot.T), self.X_train))

        pivot = pv.pivot('time_id','stock_id','book_log_return1_mean')
        pivot = pivot.fillna(pivot.mean())
        pivot = pd.DataFrame(minmax_scale(pivot))
        stock_neighbor.append(StockNeighbors('stock_log_return1_mean', 8, minmax_scale(pivot.T), self.X_train))

        self.df_nn = self.X_train.copy()
        
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
        stock_n = [20,40,60,80]
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
        self.df_nn = self.df_nn.fillna(self.df_nn.mean())
        del pv
        del self.X_train
        self.next(self.prepare_train_and_test_dataset)

    @step
    def prepare_train_and_test_dataset(self):
        """
        Prepare training and testing dataset
        """
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
        self.X = self.df_train[[col for col in self.df_train.columns if col not in ['target','time_id','stock_id','level_0','index']]]
        self.y = self.df_train['target']
        self.X_test = self.df_test[[col for col in self.df_test.columns if col not in ['target','time_id','stock_id','level_0','index']]]
        self.y_test = self.df_test['target']
        print(self.y.head())
        
        del self.df_nn
        self.next(self.set_params)
    @step
    def set_params(self):
        """
        Tune hyperparameters
        """
        self.params = [0.01, 0.2]
        self.next(self.train_model, foreach='params')
    @step
    def train_model(self):
        """
        Train a model on the training set
        """
        import lightgbm as lgb
        experiment.add_tag("learning_rate=" + str(self.input))
        experiment.add_tag("train")
        self.param = {
            'objective': 'regression',
            'verbose': -1,
            'metric': 'None',
            'reg_alpha': 5,
            'reg_lambda': 5,
            'min_data_in_leaf': 256,
            'max_depth': 12,
            'num_leaves': 800,
            'colsample_bytree': 0.3,
            'learning_rate': 0.01
            }
        self.param['learning_rate'] = self.input
        experiment.log_parameter("param", self.param)
        print(self.param)
        
        from sklearn.model_selection import KFold
        self.model = None
        self.best_RMSPE = 2**31
        self.X_train, self.X_val = self.X.iloc[self.fold[3][0]], self.X.iloc[self.fold[3][1]]
        self.y_train, self.y_val = self.y.iloc[self.fold[3][0]], self.y.iloc[self.fold[3][1]]
        # Root mean squared percentage error weights
        train_weights = 1 / np.square(self.y_train)
        val_weights = 1 / np.square(self.y_val)
        train_dataset = lgb.Dataset(self.X_train, self.y_train, weight = train_weights)
        val_dataset = lgb.Dataset(self.X_val, self.y_val, weight = val_weights)
        self.model = lgb.train(params = self.param,
                          num_boost_round=1000,
                          train_set = train_dataset, 
                          valid_sets = [train_dataset, val_dataset], 
                          verbose_eval = 250,
                          early_stopping_rounds=50,
                          feval = feval_rmspe)
        self.next(self.validate_model)
        
    @step
    def validate_model(self):
        """
        Validate the model
        """
        experiment.add_tag("learning_rate=" + str(self.input))
        experiment.add_tag("validation")
        ypred = self.model.predict(self.X_val)
        rmspe_score = rmspe(self.y_val, ypred)
        self.train_val_result = {"RMSPE": rmspe_score, "model": self.model,"param":self.param, "X_test": self.X_test, "y_test": self.y_test, "df_test": self.df_test}
        self.metrics = {"param":self.param, "RMSPE": rmspe_score}
        # log metrics to experiment
        experiment.log_metrics(self.metrics)
        self.next(self.join)

    @step
    def join(self, inputs):
        """
        Join the result, and select hyperparameters that generate the best result
        """
        experiment.add_tag("join")
        self.train_val_results = [input.train_val_result for input in inputs]
        # select the best model
        self.best_model = min(self.train_val_results, key=lambda train_val_result: train_val_result["RMSPE"])
        self.best_param = self.best_model["param"]
        self.best_lgb_model = self.best_model["model"]
        self.X_test = self.best_model['X_test']
        self.y_test = self.best_model['y_test']
        self.df_test = self.best_model['df_test']
        # log all metrics in order to help selecting models from a visual perspective
        for input in inputs:
            experiment.log_metric("learning_rate=%s_rmspe" % input.train_val_result["param"]["learning_rate"], input.train_val_result["RMSPE"])
        experiment.log_parameter("Best Model", self.best_param)
        self.next(self.test_model)
    @step 
    def test_model(self):
        """
        Test the model on the hold out sample
        """
        experiment.add_tag("test")
        self.df_test_group_by_stock = self.df_test.groupby('stock_id')
        self.y_pred = self.best_lgb_model.predict(self.X_test, num_iteration=self.best_lgb_model.best_iteration)
        self.best_model_rmspe = rmspe(self.y_test, self.y_pred)
        print(f"# RMSPE: {self.best_model_rmspe}")   
        self.metrics = {"param": self.best_param, "RMSPE": self.best_model_rmspe}
        experiment.log_metrics(self.metrics)
        self.next(self.end)

    @step
    def end(self):
        # all done, just print goodbye
        print("All done at {}!\n See you, space cowboys!".format(datetime.utcnow()))


if __name__ == '__main__':
    MyFlow()
