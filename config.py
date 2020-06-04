import yaml
from typing import NamedTuple, List, Union, Dict

class Config(object):
  def __init__(self, path: str = 'config.yml'):
    self._path = path
    self.read_config()

  def read_config(self) -> None:
    with open(self._path) as file:
      self._config = yaml.load(file, Loader=yaml.FullLoader)

  def add_arm(self, arm: str, value: str) -> None:
    self._config['arms'][arm] = value

  @property
  def arms(self) -> dict:
    return self._config['arms']