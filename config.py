import yaml

class Config(object):
  def __init__(self, path = 'config.yml'):
    self._path = path
    self.read_config()

  def read_config(self):
    with open(self._path) as file:
      self._config = yaml.load(file, Loader=yaml.FullLoader)

  def add_arm(self, arm, value):
    self._config['arms'][arm] = value

  @property
  def arms(self):
    return self._config['arms']