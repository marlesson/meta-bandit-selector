import sys, os
#sys.path.insert(0, os.path.dirname(__file__))
#sys.path.insert(0, "/media/workspace/DeepFood/ifood_cvae_model/src")

import unittest
import pandas as pd
from unittest.mock import Mock
from lib.model_control import *
from config import Config

class TestModelControl(unittest.TestCase):

  def setUp(self):
      self.config = Config('tests/src/factories/config/config.yml')
      
      
  def test_update(self):
    model  = ModelControl(self.config)

    c = {
      'f1': 0,
      'f2': 1
    }

    a = 'arm1'
    
    result = model.update(c, a, 1)

    self.assertEqual(result.get(), 1.0)

  def test_predict_proba(self):
    model  = ModelControl(self.config)

    dataset = [
        {'x': {'f1': 0, 'f2': 1}, 'y': 'arm1'},
        {'x': {'f1': 1, 'f2': 0}, 'y': 'arm2'},
        {'x': {'f1': 1, 'f2': 1}, 'y': 'arm3'}
      ]

    for row in dataset:
      model.update(row['x'], row['y'], 1)

    c = {'f1': 0, 'f2': 0}
    predict = model.predict_proba(c)        

  def test_select_arm(self):
    model  = ModelControl(self.config)

    dataset = [
        {'x': {'f1': 0, 'f2': 1}, 'y': 'arm1'},
        {'x': {'f1': 1, 'f2': 0}, 'y': 'arm2'},
        {'x': {'f1': 1, 'f2': 1}, 'y': 'arm3'}
      ]

    for row in dataset:
      model.update(row['x'], row['y'], 1)

    predict = model.select_arm({'f1': 1, 'f2': 0})        
    self.assertEqual(predict, "arm2")

    

if __name__ == '__main__':
    unittest.main()