
from lib.model_control import *
from config import Config

class MetaBandit(object):

  def __init__(self, config, policy, arms):
    self._config = config
    self._policy = policy
    self._arm    = arms

  def update(self, input):
    context = input["context"]
    arm     = input["arm"]
    reward  = int(input["reward"])

    self._policy.update(context, arm, reward)

  def predict(self, input):

    context_input = input["context"]
    model_input   = input["input"]

    arm    = self._policy.select_arm(context_input)
    result = self._arm.request(arm, model_input)

    return {
      "result": result,
      "bandit": {
        "arm": arm
      }
    }
