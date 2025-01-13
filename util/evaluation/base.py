from abc import ABC

class BaseEvaluator(ABC):
    def __init__(self, eval_frequency: int = 10):
        self.eval_frequency = eval_frequency
        self.metrics = {}
