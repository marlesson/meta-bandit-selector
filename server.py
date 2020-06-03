from bentoml import env, artifacts, api, BentoService
from bentoml.handlers import JsonHandler
from bentoml.artifact import PickleArtifact

@env(auto_pip_dependencies=True)
@artifacts([PickleArtifact('model')])
class MetaBanditClassifier(BentoService):

    @api(JsonHandler)
    def update(self, input):
        return self.artifacts.model.update(input)

    @api(JsonHandler)
    def predict(self, input):
        return self.artifacts.model.predict(input)

