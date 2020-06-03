import argparse
import responses

from server import MetaBanditClassifier 
from lib.meta_bandit import MetaBandit
import importlib
import json
from config import Config

def get_policy(args):
  polity_module = importlib.import_module(args.polity_module)
  policy_class  = getattr(polity_module, args.polity_cls)
  return policy_class(config)

@responses.activate
def test_server(server):
  responses.add(responses.POST, 'http://arm1.localhost.com',
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


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='Process some integers.')
  parser.add_argument('--config-path', help='')
  parser.add_argument('--polity-module', help='')
  parser.add_argument('--polity-cls', help='')

  args = parser.parse_args()
  
  print(args)

  config = Config(args.config_path)
  policy = get_policy(args)

  meta_bandit = MetaBandit(config, policy)
  
  meta_bandit_server = MetaBanditClassifier()
  meta_bandit_server.pack("model", meta_bandit)

  meta_bandit_server.save()

  test_server(meta_bandit_server)

  #server_classifier = server()

