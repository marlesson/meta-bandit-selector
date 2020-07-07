from bentoml import BentoService, config as bentoml_config
from prometheus_client import Histogram, Counter, Gauge, Summary
from prometheus_client.context_managers import ExceptionCounter
from config import Config
_DEFAULT_ENDPOINT = "predict"


class MetaBanditMonitor(object):
    def __init__(self, bento_service: BentoService, config: Config) -> None:
        self._version = bento_service.version

        service_name = bento_service.name
        namespace = bentoml_config("instrument").get("default_namespace")

        self._metric = Summary(
            name=service_name + "_oracle_metric",
            documentation=" Oracle Metric",
            namespace=namespace,
            labelnames=["endpoint", "service_version"],
        )

        self._selected_arm = Counter(name=service_name + "_arm_total", 
                                          documentation='Total number selected arm',
                                          namespace=namespace,
                                          labelnames=["endpoint", "service_version", "arm"])

    def observe_metric_value(self, value: float, endpoint: str = _DEFAULT_ENDPOINT):
      self._metric.labels(endpoint, self._version).observe(value)

    def observe_selected_arm(self, arm: str, endpoint: str = _DEFAULT_ENDPOINT):
      self._selected_arm.labels(endpoint, self._version, arm).inc()