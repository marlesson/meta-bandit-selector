import requests
import json
from config import Config

class ArmControl(object):
  
  def __init__(self, config: Config):
    self._config = config

  def request(self, arm: str, payload: dict) -> dict:
    endpoint = self._config.arms[arm]

    r = requests.post(endpoint, 
                      data=json.dumps(payload), 
                      headers={"Content-Type": "application/json"} )
    
    r = json.loads(r.text)
    
    return r


# import requests
# try: 
#     url = "http://google.com"
#     r = requests.get(url, timeout=10)
# except requests.exceptions.Timeout as e: 
#     print e    