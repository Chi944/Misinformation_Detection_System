import numpy as np

from src.utils.logger import get_logger


class TemperatureScaler:
    """
    Post-hoc probability calibration using temperature scaling.

    Learns a single temperature parameter T that scales logits so that
    the model's confidence better matches empirical accuracy.

    Calibration should be run on a held-out validation set AFTER
    the main model is trained, not during training.

    Args:
        init_temp (float): starting temperature. Default 1.0 (no-op)
    """

    def __init__(self, init_temp=1.0):
        self.temperature = float(init_temp)
        self.logger = get_logger(__name__)

    def fit(self, logits, y_true, lr=0.01, max_iter=50):
        """
        Fit temperature by minimising NLL on validation logits.

        Args:
            logits (array): raw pre-sigmoid logits, shape (N,)
            y_true (array): ground truth labels 0/1, shape (N,)
            lr (float): learning rate for gradient descent
            max_iter (int): maximum gradient steps
        Returns:
            float: fitted temperature
        """
        logits = np.array(logits, dtype=float)
        y_true = np.array(y_true, dtype=float)
        T = self.temperature

        for step in range(max_iter):
            scaled = logits / T
            probs = self._sigmoid(scaled)
            probs = np.clip(probs, 1e-7, 1 - 1e-7)
            nll = -np.mean(y_true * np.log(probs) + (1 - y_true) * np.log(1 - probs))
            # Gradient of NLL w.r.t. T
            grad = np.mean((probs - y_true) * (-logits / (T**2)))
            T = T - lr * grad
            T = max(0.01, T)  # prevent collapse to zero

            if step % 10 == 0:
                self.logger.debug("Calibration step %d T=%.4f NLL=%.4f", step, T, nll)

        self.temperature = T
        self.logger.info("Temperature calibration done: T=%.4f", T)
        return T

    def calibrate(self, probs):
        """
        Apply temperature scaling to raw probabilities.

        Converts probs -> logits -> scale by T -> convert back.

        Args:
            probs (array): raw model probabilities in [0, 1]
        Returns:
            array: calibrated probabilities
        """
        probs = np.clip(np.array(probs, dtype=float), 1e-7, 1 - 1e-7)
        logits = np.log(probs / (1 - probs))  # inverse sigmoid
        scaled = logits / self.temperature
        return self._sigmoid(scaled)

    def calibrate_single(self, prob):
        """
        Calibrate a single probability value.

        Args:
            prob (float): raw probability
        Returns:
            float: calibrated probability
        """
        return float(self.calibrate(np.array([prob]))[0])

    def _sigmoid(self, x):
        """Numerically stable sigmoid."""
        return np.where(x >= 0, 1.0 / (1.0 + np.exp(-x)), np.exp(x) / (1.0 + np.exp(x)))

    def get_state(self):
        """Return serialisable state dict."""
        return {"temperature": self.temperature}

    def load_state(self, state):
        """
        Load temperature from state dict.

        Args:
            state (dict): previously saved state from get_state()
        """
        self.temperature = float(state.get("temperature", 1.0))


class EnsembleCalibrator:
    """
    Calibrates all 3 models plus the ensemble using TemperatureScaler.

    Maintains one TemperatureScaler per model. Calibrated probabilities
    are fed back into ensemble weight recalibration.

    Args:
        None
    """

    def __init__(self):
        self.scalers = {
            "bert": TemperatureScaler(),
            "tfidf": TemperatureScaler(),
            "naive_bayes": TemperatureScaler(),
            "ensemble": TemperatureScaler(),
        }
        self.logger = get_logger(__name__)

    def fit_all(self, model_logits_dict, y_true):
        """
        Fit a temperature scaler for each model simultaneously.

        Args:
            model_logits_dict (dict): model_name -> logits array
            y_true (array): ground truth labels
        Returns:
            dict: model_name -> fitted temperature value
        """
        temperatures = {}
        for name, logits in model_logits_dict.items():
            if name not in self.scalers:
                self.scalers[name] = TemperatureScaler()
            T = self.scalers[name].fit(logits, y_true)
            temperatures[name] = T
            self.logger.info("Calibrated %s T=%.4f", name, T)
        return temperatures

    def calibrate_probs(self, model_probs_dict):
        """
        Calibrate probabilities for all models.

        Args:
            model_probs_dict (dict): model_name -> probs array or float
        Returns:
            dict: model_name -> calibrated probs
        """
        calibrated = {}
        for name, probs in model_probs_dict.items():
            if name in self.scalers:
                calibrated[name] = self.scalers[name].calibrate(np.atleast_1d(probs))
            else:
                calibrated[name] = probs
        return calibrated

    def get_state(self):
        """Return serialisable state for all scalers."""
        return {name: scaler.get_state() for name, scaler in self.scalers.items()}

    def load_state(self, state):
        """
        Load scaler states from dict.

        Args:
            state (dict): previously saved state from get_state()
        """
        for name, s in state.items():
            if name not in self.scalers:
                self.scalers[name] = TemperatureScaler()
            self.scalers[name].load_state(s)
