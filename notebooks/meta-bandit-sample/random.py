import bentoml
import pandas as pd
import numpy as np
import random
from bentoml.artifact import PickleArtifact
from bentoml.handlers import JsonHandler

@bentoml.ver(1, 0)
class RandomRecommender(bentoml.BentoService):

    @bentoml.api(JsonHandler)
    def rank(self, sample):
        articles = sample['Article_List']
        scores = [random.random() for art in articles]
        output = [item for score, item in sorted(zip(scores, articles),reverse=True)]
        
        return {
            "articles": output,
            "scores": sorted(scores, reverse=True)
        }
