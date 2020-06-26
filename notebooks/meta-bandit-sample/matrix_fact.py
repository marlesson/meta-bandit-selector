import bentoml
import pandas as pd
import numpy as np

from bentoml.artifact import PickleArtifact
# from bentoml.adapters import DataframeInput
from bentoml.handlers import DataframeHandler

@bentoml.ver(1, 0)
@bentoml.artifacts([
    PickleArtifact("index_map"),
    PickleArtifact("cluster_path"),
    PickleArtifact("matrix"),
])


class ClusteredMatrixFactRecommender(bentoml.BentoService):

    def get_index(self, item):
        if item in self.artifacts.index_map:
            return self.artifacts.index_map[item]
        else:
            return 0


    def get_score(self, index, cluster):
        if index is 0:
            return -1
        else:
            return self.artifacts.matrix[index-1, cluster]

    @bentoml.api(DataframeHandler)
    def rank(self, sample):
        articles = sample['Article_List']
        indexed_articles = [self.get_index(art) for art in articles]
        user_cluster = self.artifacts.cluster_path.predict([sample['User_Features']])[0]
        scores = [self.get_score(art, user_cluster) for art in indexed_articles]
        output = [item for score, item in sorted(zip(scores, articles),reverse=True)]
        return output
