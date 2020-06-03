
from lib.model_control import *
from lib.arm_control import ArmControl
from config import Config

class MetaBandit(object):

  def __init__(self, config, policy):
    self._config = config
    self._policy = policy
    self._arm    = ArmControl(config)

  def update(self, input):
    context = input["context"]
    arm     = input["arm"]
    reward  = int(input["reward"])

    self._policy.update(context, arm, reward)

  def predict(self, input):

    model_input   = input["input"]
    context_input = input["context"]

    arm    = self._policy.select_arm(context_input)
    result = self._arm.request(arm, model_input)

    return {
      "result": result,
      "bandit": {
        "arm": arm
      }
    }
