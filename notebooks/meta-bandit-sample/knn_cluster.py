import bentoml
import pandas as pd
import numpy as np

from bentoml.artifact import PickleArtifact
# from bentoml.adapters import DataframeInput
from bentoml.handlers import DataframeHandler
from bentoml.handlers import JsonHandler

@bentoml.ver(1, 0)
@bentoml.artifacts([
    PickleArtifact("knn"),
    PickleArtifact("index_map"),
    PickleArtifact("cluster_path"),
    PickleArtifact("pop_matrix"),
])


class ClusteredKNN(bentoml.BentoService):

    def get_index(self, item):
        if item in self.artifacts.index_map:
            return self.artifacts.index_map[item]
        else:
            return 0

    def setup_scores(self, features, n_neighbors):
        neighbors_idxs = self.artifacts.knn.kneighbors(X=features, n_neighbors=n_neighbors, return_distance=False) # get indexes of neighbors
        knclusters = self.artifacts.cluster_path.labels_[neighbors_idxs] # get clusters of neighbors
        clicks = [self.artifacts.pop_matrix[c] for c in knclusters] # create an array with the number of item iteractions per cluster (per item)
        clicks = np.asarray(clicks[0])
        self.mean_scores = np.mean(clicks, axis=0) # mean over the number of iteractions to create a weighted score


    def get_score(self, index):
        if index is 0:
            return -1
        else:
            return self.mean_scores[index]

    @bentoml.api(JsonHandler)
    def rank(self, sample):
        n_neighbors = 50
        articles = sample['Article_List']
        indexed_articles = [self.get_index(art) for art in articles]
        user_features = sample['User_Features']
        self.setup_scores(np.asarray([user_features]), n_neighbors)
        scores = [self.get_score(idx) for idx in indexed_articles]
        output = [item for score, item in sorted(zip(scores, articles),reverse=True)]

        return {
            "articles": output,
            "scores": sorted(scores, reverse=True)
        }
