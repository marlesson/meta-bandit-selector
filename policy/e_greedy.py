from lib.model_control import ModelControl


class EGreedyPolicy(ModelControl):
    def __init__(self, config = None, epsilon: float = 0.1, seed: int = 42) -> None:
      super().__init__(config, seed)
      self._epsilon = epsilon
  
    def select_arm(self, context):
      
      if self._rng.choice([True, False], p=[self._epsilon, 1.0 - self._epsilon]):
        # Select random endpoint
        arm = self._rng.choice(list(self._config.arms.keys()))
      else: 
        # Select best endpoint
        arm = super().select_arm(context)

      return arm