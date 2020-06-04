
from creme import compose
from creme import linear_model, multiclass
from creme import metrics
from creme import preprocessing
from creme import optim
from creme import sampling
from numpy.random.mtrand import RandomState
import numpy as np
from config import Config

class ModelControl(object):

  def __init__(self, config: Config = None, seed: int = 42):
    self._config  = config
    self._rng     = RandomState(seed)
    self._oracle  = self.build_oracle()
    self._oracle_metric  = metrics.MacroF1()
    self._times   = 1
    self._arms_selected = []
    self._arms = list(self._config.arms.keys())
    self.init_default_reward()
  
  def init_default_reward(self):
    for a in self._arms:
      self.update({}, a, 1)

  def build_oracle(self) -> compose.Pipeline:
    model = compose.Pipeline(
        ('scale', preprocessing.StandardScaler()),
        ('learn', multiclass.OneVsRestClassifier(
            binary_classifier=linear_model.LogisticRegression())
        )
    )        
    return model

  def update(self, context: dict, arm: str, reward: int) -> None:
    if reward and arm in self._arms:
      self._oracle.fit_one(context, arm)
    return True

  def predict_proba(self, context: dict) -> dict:
    return self._oracle.predict_proba_one(context)

  def select_arm(self, context: dict) -> str:
    pred = self.predict_proba(context)
    arm  = self._arms[np.argmax(list(pred.values()))]

    return arm