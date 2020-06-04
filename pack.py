import argparse
import responses

from server import MetaBanditClassifier 
from lib.meta_bandit import MetaBandit
from lib.arm_control import ArmControl
import importlib
import json
from config import Config

def get_policy(config: Config, args):
  polity_module = importlib.import_module(args.polity_module)
  policy_class  = getattr(polity_module, args.polity_cls)

  params = {"config": config}
  extra_params = config._config['bandit_policy_params']
  return policy_class(**{**params, **extra_params})

@responses.activate
def test_server(config: Config, server: MetaBanditClassifier):
  # mock Request
  responses.add(responses.POST,  list(config.arms.values())[2],
      json.dumps({}),
      headers={'content-type': 'application/json'},
  )

  payload = {
    "context": {
      "f1": 1,
      "f2": 0
    },
    "input": {
      "user": 1,
      "items": [
        0,
        1,
        3,
        7,
        4,
        6,
        5,
        2
      ]
    }
  }

  print(server.predict(payload))

# python pack.py --config-path config.yml --polity-module policy.e_greedy --polity-cls EGreedyPolicy
# bentoml serve MetaBanditClassifier:latest
if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='Process some integers.')
  parser.add_argument('--config-path', help='')
  parser.add_argument('--polity-module', help='')
  parser.add_argument('--polity-cls', help='')

  args = parser.parse_args()
  
  print(args)

  # Build Model
  config       = Config(args.config_path)
  arm_control  = ArmControl(config)
  policy_control = get_policy(config, args)

  meta_bandit = MetaBandit(config, policy_control, arm_control)
  
  # Package Model
  meta_bandit_server = MetaBanditClassifier()
  meta_bandit_server.pack("model", meta_bandit)
  meta_bandit_server.save()

  # Test Model
  test_server(config, meta_bandit_server)
