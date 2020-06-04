from bentoml import env, artifacts, api, BentoService
from bentoml.handlers import JsonHandler
from bentoml.artifact import PickleArtifact
from lib.meta_bandit import MetaBandit
from policy import e_greedy, softmax
from lib.monitoring import MetaBanditMonitor

import bentoml

@env(auto_pip_dependencies=True)
@bentoml.ver(1, 0)
@artifacts([PickleArtifact('model')])
class MetaBanditClassifier(BentoService):

    @property
    def monitor(self) -> MetaBanditMonitor:
        if not hasattr(self, "_monitor"):
            self._monitor = MetaBanditMonitor(self, self.model._config)
        return self._monitor

    @property
    def model(self) -> MetaBandit:
        if not hasattr(self, "_model"):
            self._model: MetaBandit = self.artifacts.model
        return self._model

    @api(JsonHandler)
    def update(self, input: dict) -> dict:
        r = self.model.update(input)
        self.monitor.observe_metric_value(r['metric'], 'update')

        return r

    @api(JsonHandler)
    def predict(self, input: dict) -> dict:
        r = self.model.predict(input)
        self.monitor.observe_selected_arm(r['bandit']['arm'])

        return r

