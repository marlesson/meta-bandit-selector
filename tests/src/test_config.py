import sys, os
#sys.path.insert(0, os.path.dirname(__file__))
#sys.path.insert(0, "/media/workspace/DeepFood/ifood_cvae_model/src")

import unittest
import pandas as pd
from unittest.mock import Mock
from config import Config

class TestConfig(unittest.TestCase):

  def test_read_arms_config(self):
    config = Config('tests/src/factories/config/config.yml')

    self.assertEqual(len(config.arms), 2)

  def test_add_arm_config(self):
    config = Config('tests/src/factories/config/config.yml')

    config.add_arm('arm3', 'localhost')

    self.assertEqual(len(config.arms), 3)

if __name__ == '__main__':
    unittest.main()