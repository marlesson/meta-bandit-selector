import sys, os
#sys.path.insert(0, os.path.dirname(__file__))
#sys.path.insert(0, "/media/workspace/DeepFood/ifood_cvae_model/src")

import unittest
import pandas as pd
from unittest.mock import Mock
from lib.meta_bandit import *
from policy.e_greedy import EGreedyPolicy
from config import Config
import responses
import requests
import json
from requests.exceptions import ConnectionError


class TestMetaBandit(unittest.TestCase):

  def setUp(self):
    self.config = Config('tests/src/factories/config/config.yml')
    self.policy = EGreedyPolicy(self.config)
  @property
  def payload(self):
    with open('tests/src/factories/requests/main_payload.json') as json_file:
      data = json.load(json_file)

    return data

  @property
  def payload_result(self):
    with open('tests/src/factories/requests/payload_result.json') as json_file:
      data = json.load(json_file)

    return data

  @property
  def payload_update(self):
    with open('tests/src/factories/requests/payload_update.json') as json_file:
      data = json.load(json_file)

    return data

  @responses.activate
  def test_predict(self):
    responses.add(responses.POST, 'http://arm1.localhost.com',
        json.dumps(self.payload_result),
        headers={'content-type': 'application/json'},
    )

    meta_bandit = MetaBandit(self.config, self.policy)

    result      = meta_bandit.predict(self.payload)

    self.assertEqual(result['result'], self.payload_result)
    self.assertEqual(result['bandit']['arm'], 'arm1')

  def test_update(self):
    meta_bandit = MetaBandit(self.config, self.policy)

    result      = meta_bandit.update(self.payload_update)
    print(result)

if __name__ == '__main__':
    unittest.main()