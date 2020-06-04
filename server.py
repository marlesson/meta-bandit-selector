from bentoml import env, artifacts, api, BentoService
from bentoml.handlers import JsonHandler
from bentoml.artifact import PickleArtifact
from lib.meta_bandit import MetaBandit
from policy import e_greedy

@env(auto_pip_dependencies=True)
@artifacts([PickleArtifact('model')])
class MetaBanditClassifier(BentoService):

    @api(JsonHandler)
    def update(self, input):
        return self.artifacts.model.update(input)

    @api(JsonHandler)
    def predict(self, input):
        return self.artifacts.model.predict(input)

