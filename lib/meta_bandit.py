
from lib.model_control import *
from config import Config
from lib.arm_control import ArmControl
from lib.model_control import ModelControl

class MetaBandit(object):

  def __init__(self, config: Config, policy: ModelControl, arms: ArmControl):
    self._config = config
    self._policy = policy
    self._arm    = arms

  def update(self, input: dict) -> None:
    context = input["context"]
    arm     = input["arm"]
    reward  = int(input["reward"])

    self._policy.update(context, arm, reward)

  def predict(self, input: dict) -> dict:

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
