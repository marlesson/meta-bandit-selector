import bentoml
import pandas as pd
import numpy as np

from bentoml.artifact import PickleArtifact
# from bentoml.adapters import DataframeInput
from bentoml.handlers import DataframeHandler

@bentoml.ver(1, 0)
@bentoml.artifacts([
    PickleArtifact("index_map"),
    PickleArtifact("item_score"),
])


class MostPopularRecommender(bentoml.BentoService):

    def get_index(self, item):
        if item in self.artifacts.index_map:
            return self.artifacts.index_map[item]
        else:
            return 0


    def get_score(self, index):
        return self.artifacts.item_score[index]

    @bentoml.api(DataframeHandler)
    def rank(self, sample):
        articles = sample['Article_List']
        indexed_articles = [self.get_index(art) for art in articles]
        scores = [self.get_score(art) for art in indexed_articles]
        output = [item for score, item in sorted(zip(scores, articles),reverse=True)]
        return output


# def pack_indexmap(bento: MostPopularRecommender, indexmap: dict):
#     bento.pack("index_map", indexmap)
#
#
# def pack_itemscore(bento: MostPopularRecommender, itemscore: dict):
#     bento.pack("item_score", itemscore)
