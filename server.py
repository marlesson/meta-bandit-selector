from bentoml import env, artifacts, api, BentoService
from bentoml.handlers import JsonHandler
from bentoml.artifact import PickleArtifact
from lib.meta_bandit import MetaBandit
from policy import e_greedy, softmax

import bentoml

@env(auto_pip_dependencies=True)
@bentoml.ver(1, 0)
@artifacts([PickleArtifact('model')])
class MetaBanditClassifier(BentoService):

    @property
    def model(self) -> MetaBandit:
        if not hasattr(self, "_model"):
            self._model: MetaBandit = self.artifacts.model
        return self._model

    @api(JsonHandler)
    def update(self, input: dict) -> dict:
        return self.model.update(input)

    @api(JsonHandler)
    def predict(self, input: dict) -> dict:
        return self.model.predict(input)

