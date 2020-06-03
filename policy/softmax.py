from lib.model_control import ModelControl
from scipy.special import softmax, expit


class SotfmaxPolicy(ModelControl):
    def __init__(self, config = None, logit_multiplier: float = 2.0, seed: int = 42) -> None:
      super().__init__(config, seed)
      self._logit_multiplier = logit_multiplier
  
    def select_arm(self, context):
      pred         = self.predict_proba(context)
      scores       = list(pred.values())
      
      scores_logit = expit(scores)
      arms_probs   = softmax(self._logit_multiplier * scores_logit)

      return self._rng.choice(list(pred.keys()), p = arms_probs)