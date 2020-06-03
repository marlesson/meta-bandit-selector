import requests
import json

class ArmControl(object):
  
  def __init__(self, config):
    self._config = config

  def request(self, arm, payload):
    endpoint = self._config.arms[arm]

    r = requests.post(endpoint, 
                      data=json.dumps(payload), 
                      headers={"Content-Type": "application/json"} )
    
    r = json.loads(r.text)
    
    return r