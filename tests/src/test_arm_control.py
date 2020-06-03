import sys, os
#sys.path.insert(0, os.path.dirname(__file__))
#sys.path.insert(0, "/media/workspace/DeepFood/ifood_cvae_model/src")

import unittest
import pandas as pd
from unittest.mock import Mock
from lib.arm_control import *
from config import Config
import responses
import requests
import json
from requests.exceptions import ConnectionError


class TestArmControl(unittest.TestCase):

  def setUp(self):
    self.config = Config('tests/src/factories/config/config.yml')
  
  @property
  def payload(self):
    with open('tests/src/factories/requests/payload.json') as json_file:
      data = json.load(json_file)

    return data

  @property
  def payload_result(self):
    with open('tests/src/factories/requests/payload_result.json') as json_file:
      data = json.load(json_file)

    return data

  @responses.activate
  def test_request(self):
    responses.add(responses.POST, 'http://arm1.localhost.com',
        json.dumps(self.payload_result),
        headers={'content-type': 'application/json'},
    )

    arm_control = ArmControl(self.config)

    result = arm_control.request('arm1', self.payload)

    self.assertEqual(result, self.payload_result)

  # @responses.activate
  # def test_request_fail(self):
  #   # responses.add(responses.POST, 'http://arm1.localhost.com',
  #   #     json.dumps(self.payload_result),
  #   #     headers={'content-type': 'application/json'},
  #   # )

  #   arm_control = ArmControl(self.config)

  #   #with pytest.raises(ConnectionError):
  #   result = arm_control.request('arm1', self.payload)

  #   self.assertEqual(result, self.payload_result)


if __name__ == '__main__':
    unittest.main()