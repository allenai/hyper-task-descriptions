"""
T5X default setup does not respect the learning rate set in its arguments.
This optimizer version does.
"""
from t5x import adafactor


class Adafactor(adafactor.Adafactor):
    def apply_param_gradient(self, step, hyper_params, param, state, grad, path):
        # must use replace function as hyper_params is a struct.dataclass and frozen
        hyper_params = hyper_params.replace(learning_rate=self.hyper_params.learning_rate)
        return super().apply_param_gradient(step, hyper_params, param, state, grad, path)

    def apply_gradient(self, hyper_params, params, state, grads):
        # must use replace function as hyper_params is a struct.dataclass and frozen
        hyper_params = hyper_params.replace(learning_rate=self.hyper_params.learning_rate)
        return super().apply_gradient(hyper_params, params, state, grads)
