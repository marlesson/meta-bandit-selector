
from creme import compose
from creme import linear_model, multiclass
from creme import metrics
from creme import preprocessing
from creme import optim
from creme import sampling
from numpy.random.mtrand import RandomState
import numpy as np

class ModelControl(object):

  def __init__(self, config = None, seed = 42):
    self._config  = config
    self._rng     = RandomState(seed)
    self._oracle  = self.build_oracle()
    self._oracle_metric  = metrics.MacroF1()
    self._times   = 1
    self.init_default_reward()
  
  def init_default_reward(self):
    for a, v in self._config.arms.items():
      self.update({}, a, 1)

  def build_oracle(self):
    model = compose.Pipeline(
        ('scale', preprocessing.StandardScaler()),
        ('learn', multiclass.OneVsRestClassifier(
            binary_classifier=linear_model.LogisticRegression())
        )
    )        
    return model

  def update(self, context, arm, reward):
    if reward:
      self._oracle.fit_one(context, arm)
    return True

  def predict_proba(self, context):
    return self._oracle.predict_proba_one(context)

  def select_arm(self, context):
    pred = self.predict_proba(context)
    arm  = list(pred.keys())[np.argmax(pred.values())]

    return arm