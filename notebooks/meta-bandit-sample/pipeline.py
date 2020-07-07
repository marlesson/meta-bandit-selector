import re
import ast
import luigi
import pandas as pd
import numpy as np
import datetime
import os
import bentoml
from sklearn.externals import joblib
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds

from .preprocessing import preprocess, read_sample
from .knn_cluster import ClusteredKNN
from .most_popular import MostPopularRecommender
from .random import RandomRecommender
from .matrix_fact import ClusteredMatrixFactRecommender
from .rank import ndcg_at_k, precision_at_k

DATASET_DIR = "dataset"
FILE_DATASET = "/media/backup/datasets/yahoo/yahoo_dataset_clicked.csv"


def literal_eval(element):
    if isinstance(element, str):
        return ast.literal_eval(re.sub('\s+', ',', element))
    return element

def test_rank(model):
    test_articles = [11, 565648, 563115, 552077, 564335, 565589, 563938, 560290, 563643, 560620, 565822, 563787, 555528, 565364, 559855, 560518]  
    return model.rank({'User_Features': [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 
                                        1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                        0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                        0, 0, 0, 0], 
                                        'Article_List': np.asarray(test_articles)})

class PrepareDataset(luigi.Task):
    split_test: float = luigi.FloatParameter(default=0.1)
    sample_train: float = luigi.FloatParameter(default=0.1)

    def output(self):
        return (luigi.LocalTarget(
          os.path.join(DATASET_DIR, "train_%2f_%2f.csv" % (self.split_test, self.sample_train))),
          luigi.LocalTarget(os.path.join(DATASET_DIR, "test_%2f_%2f.csv" % (self.split_test, self.sample_train))))

    def transform(self, df):
      df['User_Features'] = df['User_Features'].apply(literal_eval)
      df['Article_List']  = df['Article_List'].apply(literal_eval)
      
      return df

    def time_train_test_split(
        self, df: pd.DataFrame, test_size: float, timestamp_property: str
    ):
        df = df.sort_values(timestamp_property)
        size = len(df)
        cut = int(size - size * test_size)

        return df.iloc[:cut], df.iloc[cut:]

    def run(self):
        os.makedirs(DATASET_DIR, exist_ok=True)

        df = read_sample(FILE_DATASET, p=self.sample_train)
        df = self.transform(df)
        print(df)

        train, test = self.time_train_test_split(
            df, self.split_test, "Timestamp")
        
        train.to_csv(os.path.join(self.output()[0].path))
        test.to_csv(os.path.join(self.output()[1].path))

class TrainMostPopularRecommender(luigi.Task):
    split_test: float = luigi.FloatParameter(default=0.1)
    sample_train: float = luigi.FloatParameter(default=0.1)

    def requires(self):
        return PrepareDataset(split_test=self.split_test, sample_train=self.sample_train)

    def output(self):
        return luigi.LocalTarget(
                os.path.join(DATASET_DIR, "TrainMostPopularRecommender_%2f_%2f" % (self.split_test, self.sample_train)))

    def get_index_mag(self, df):
        index_map = {}
        idx = 1 # idx starts at 1 so that 0 is used for when the article is not found in the index map
        for art in df['Clicked_Article'].unique():
            index_map[art] = idx
            idx+=1

        return index_map

    def run(self):
        os.makedirs(self.output().path, exist_ok=True)

        df = pd.read_csv(self.input()[0].path)
        
        articles  = df['Clicked_Article'].unique()
        index_map = self.get_index_mag(df)

        popular = df.loc[(df['Click']==1)].groupby('Clicked_Article').size().sort_values(ascending=False)
        
        item_score = {0: -1} 
        #since 0 is used for when the article was not found in the index map, here it'll have the lowest value
        for art in articles:
            item_score[index_map[art]] = popular[art] 

        model = MostPopularRecommender()
        model.pack("item_score", item_score)
        model.pack("index_map", index_map)

        print(test_rank(model))
        model.save()
        model.save_to_dir(self.output().path)

class TrainClusteredMatrixFactRecommender(luigi.Task):
    split_test: float = luigi.FloatParameter(default=0.1)
    sample_train: float = luigi.FloatParameter(default=0.1)
    n_clusters: int = luigi.IntParameter(default=10)
    n_factors: int = luigi.IntParameter(default=5)

    def requires(self):
        return PrepareDataset(split_test=self.split_test, sample_train=self.sample_train)

    def output(self):
        return luigi.LocalTarget(
            os.path.join(DATASET_DIR, "TrainClusteredMatrixFactRecommender_%2f_%2f_%d_%d" % (self.split_test, self.sample_train, self.n_clusters, self.n_factors)))

    def get_index_mag(self, df):
        index_map = {}
        idx = 1 # idx starts at 1 so that 0 is used for when the article is not found in the index map
        for art in df['Clicked_Article'].unique():
            index_map[art] = idx
            idx+=1

        return index_map

    def get_cluster_users(self, df, n_clusters):
        users  = np.asarray(df.loc[:, 'User_Features'])  # acquire only the features
        # stack them to make an array (iteractions, features)
        users  = np.stack(users, axis=0)
        kmeans = KMeans(n_clusters=n_clusters, n_jobs=-1)
        kmeans.fit(users)
        
        return kmeans
    
    def get_matrix_score(self, df):

        pivot_table = df.pivot_table(
            index='Cluster', columns='Clicked_Article', values='Click', aggfunc=np.sum, fill_value=0)

        pivot_matrix = np.asarray(pivot_table.values, dtype='float')
        clusters     = list(pivot_table.index)

        sparse_matrix = csr_matrix(pivot_matrix)
        U, sigma, Vt  = svds(sparse_matrix, k=self.n_factors)
        sigma = np.diag(sigma)

        all_predicted_ratings = np.dot(np.dot(U, sigma), Vt)
        all_predicted_norm = (all_predicted_ratings - all_predicted_ratings.min()) / \
            (all_predicted_ratings.max() - all_predicted_ratings.min())
        cf_preds_df = pd.DataFrame(
            all_predicted_norm, columns=pivot_table.columns, index=clusters).transpose()

        return cf_preds_df

    def run(self):
        os.makedirs(self.output().path, exist_ok=True)

        df = pd.read_csv(self.input()[0].path)
        df['User_Features'] = df['User_Features'].apply(ast.literal_eval)
        df['Article_List']  = df['Article_List'].apply(ast.literal_eval)

        index_map = self.get_index_mag(df)

        # Cluster
        kmeans = self.get_cluster_users(df, self.n_clusters)
        df['Cluster'] = kmeans.labels_
        df['Clicked_Article'].replace(index_map, inplace=True)

        # Get Matrix Score
        cf_preds_df = self.get_matrix_score(df)
        matrix = np.asarray(cf_preds_df.values, dtype='float')

        # Pack
        model = ClusteredMatrixFactRecommender()
        model.pack("index_map", index_map)
        model.pack("cluster_path", kmeans)
        model.pack("matrix", matrix)

        print(test_rank(model))
        model.save()
        model.save_to_dir(self.output().path)

class TrainClusteredKNNRecommender(luigi.Task):
    split_test: float = luigi.FloatParameter(default=0.1)
    sample_train: float = luigi.FloatParameter(default=0.1)
    n_clusters: int = luigi.IntParameter(default=10)
    n_factors: int = luigi.IntParameter(default=5)

    def requires(self):
        return PrepareDataset(split_test=self.split_test, sample_train=self.sample_train)

    def output(self):
        return luigi.LocalTarget(
            os.path.join(DATASET_DIR, "TrainClusteredKNN_%2f_%2f_%d_%d" % (self.split_test, self.sample_train, self.n_clusters, self.n_factors)))

    def get_index_mag(self, df):
        index_map = {}
        idx = 1  # idx starts at 1 so that 0 is used for when the article is not found in the index map
        for art in df['Clicked_Article'].unique():
            index_map[art] = idx
            idx += 1

        return index_map

    def get_cluster_users(self, df, n_clusters):
        # acquire only the features
        users = np.asarray(df.loc[:, 'User_Features'])
        # stack them to make an array (iteractions, features)
        users = np.stack(users, axis=0)
        kmeans = KMeans(n_clusters=n_clusters, n_jobs=-1)
        kmeans.fit(users)

        return kmeans

    def get_neighbors(self, df):
        # acquire only the features
        users = np.asarray(df.loc[:, 'User_Features'])
        # stack them to make an array (iteractions, features)
        users = np.stack(users, axis=0)

        knn    = KNeighborsClassifier(n_jobs=-1)
        knn.fit(users, df['Cluster'])

        return knn

    def run(self):
        os.makedirs(self.output().path, exist_ok=True)

        df = pd.read_csv(self.input()[0].path)
        df['User_Features'] = df['User_Features'].apply(ast.literal_eval)
        df['Article_List'] = df['Article_List'].apply(ast.literal_eval)

        index_map = self.get_index_mag(df)

        # Cluster
        kmeans = self.get_cluster_users(df, self.n_clusters)
        df['Cluster'] = kmeans.labels_
        df['Clicked_Article'].replace(index_map, inplace=True)

        # Neighbors
        knn = self.get_neighbors(df)

        # Popular Matrix
        popular_by_cluster = df.loc[(df['Click'] == 1)].groupby(
            ['Cluster', 'Clicked_Article']).size().sort_values(ascending=False)

        pop_matrix = np.zeros((self.n_clusters, len(index_map)+1), dtype='int')

        for c in range(self.n_clusters):
            for i in popular_by_cluster[c].index:
                pop_matrix[c][i] = popular_by_cluster[c][i]

        # Pack
        model = ClusteredKNN()
        model.pack("index_map", index_map)
        model.pack("pop_matrix", pop_matrix)
        model.pack("knn", knn)
        model.pack("cluster_path", kmeans)

        print(test_rank(model))
        model.save()
        model.save_to_dir(self.output().path)

class TrainRandomRecommender(luigi.Task):
    def output(self):
        return luigi.LocalTarget(
            os.path.join(DATASET_DIR, "RandomRecommender"))

    def run(self):
        os.makedirs(self.output().path, exist_ok=True)

        # Pack
        model = RandomRecommender()

        print(test_rank(model))
        model.save()
        model.save_to_dir(self.output().path)

class Evaluation(luigi.Task):
    split_test: float = luigi.FloatParameter(default=0.1)
    sample_train: float = luigi.FloatParameter(default=0.1)
    n_clusters: int = luigi.IntParameter(default=10)
    n_factors: int = luigi.IntParameter(default=5)

    def requires(self):
        return (PrepareDataset(split_test=self.split_test, sample_train=self.sample_train),
                TrainMostPopularRecommender(split_test=self.split_test, sample_train=self.sample_train),
                TrainClusteredMatrixFactRecommender(
                    split_test=self.split_test, sample_train=self.sample_train, n_clusters=self.n_clusters, n_factors=self.n_factors),
                TrainClusteredKNNRecommender(split_test=self.split_test, sample_train=self.sample_train, n_clusters=self.n_clusters, n_factors=self.n_factors),
                TrainRandomRecommender())
    
    def output(self):
        return luigi.LocalTarget(
            os.path.join(DATASET_DIR, "metrics_%2f_%2f_%d_%d.csv" % (self.split_test, self.sample_train, self.n_clusters, self.n_factors)))

    def run(self):
        #os.makedirs(self.output().path, exist_ok=True)

        df = pd.read_csv(self.input()[0][1].path)
        df['User_Features'] = df['User_Features'].apply(ast.literal_eval)
        df['Article_List'] = df['Article_List'].apply(ast.literal_eval)
        result = {
                    'ndcg@5': [], 
                    'precision@1': []
                }

        print(df.head())
        results = []
        for input in self.input()[1:]:
            print(input)

            model = bentoml.load(input.path)
            model_name = type(model).__name__

            for i, row in df.iterrows():
                payload = {'User_Features': row.User_Features, 'Article_List': row.Article_List}
                output  = model.rank(payload)

                idx_clicked = output['articles'].index(row.Clicked_Article)

                r = np.zeros(len(output['articles']))
                r[idx_clicked] = 1
                
                result['ndcg@5'].append(ndcg_at_k(r, 5))
                result['precision@1'].append(precision_at_k(r, 1))

                if i % 1000 == 0:
                    print("[model_name] Evaluation...", i/len(df))
                    print(pd.DataFrame(result).mean())
                    print("")
            
            df_result = pd.DataFrame(result)#.mean()
            df_result['model'] = model_name
            results.append(df_result)
        
        df_result = pd.concat(results).groupby('model').mean()
        print(df_result)
        df_result.to_csv(self.output().path)

class EvaluationInteraction(luigi.Task):
    split_test: float = luigi.FloatParameter(default=0.1)
    sample_train: float = luigi.FloatParameter(default=0.1)
    #n_clusters: int = luigi.IntParameter(default=10)
    #n_factors: int = luigi.IntParameter(default=5)
    endpoint = 

    def requires(self):
        return PrepareDataset(split_test=self.split_test, sample_train=self.sample_train)
    
    def output(self):
        return luigi.LocalTarget(
            os.path.join(DATASET_DIR, "metrics_%2f_%2f_%d_%d.csv" % (self.split_test, self.sample_train, self.n_clusters, self.n_factors)))

    def run(self):
        # Read Test
        df = pd.read_csv(self.input()[0][1].path)
        df['User_Features'] = df['User_Features'].apply(ast.literal_eval)
        df['Article_List'] = df['Article_List'].apply(ast.literal_eval)
        
        print(df.head())
        results = []

        for i, row in df.iterrows():
            
            model_payload = {'User_Features': row.User_Features, 'Article_List': row.Article_List}
            payload = {
                "context": row.User_Features,
                "input": model_payload
            }


            if i % 1000 == 0:
                pass
        

#PYTHONPATH="." luigi --module notebooks.meta-bandit-sample.pipeline Evaluation --local-scheduler --split-test 0.2 --sample-train 1 --n-clusters 100 --n-factors 10