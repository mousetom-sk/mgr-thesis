from __future__ import annotations
from typing import Dict, List, Callable, Generator, Literal

import numpy as np
import numpy.typing as nptype

from .mlp import LMLP, LMLPLayer
from .optimization import LMLPOptimizer
from .activations import Softmax
from .metrics import Metric, Objective, SquaredError, CrossEntropy


class MLPClassifier(LMLP):

    loss: Objective = None
    labels: nptype.NDArray = None

    def __init__(self, input_dim: int, layers: List[LMLPLayer], labels: nptype.NDArray):
        super().__init__(input_dim, layers)

        if isinstance(self.layers[-1].activation, Softmax):
            self.loss = CrossEntropy()
        else:
            self.loss = SquaredError()

        self.labels = labels

    def predict_labels(self, input: nptype.NDArray) -> nptype.NDArray:
        targets = self.predict(input)

        return self.targets_to_labels(targets)
    
    def targets_to_labels(self, targets: nptype.NDArray) -> nptype.NDArray:
        label_idx = np.argmax(targets, axis=0)

        return self.labels[label_idx]
    
    def labels_to_targets(self, labels: nptype.NDArray) -> nptype.NDArray:
        if set(self.labels) != set(labels):
            raise ValueError()

        return np.array([(self.labels == l).astype(float) for l in labels]).T
    
    def train(
        self, inputs: nptype.NDArray, labels: nptype.NDArray,
        optimizer: LMLPOptimizer, metrics: Dict[str, Metric],
        max_epochs: int, batch_size: int | Literal["all"] = "all",
        epoch_end_callback: Callable[[MLPClassifier], bool] = None,
        log_freq: int | None = 5
    ):
        num_inputs = inputs.shape[1]
        targets = self.labels_to_targets(labels)
        optimizer.prepare(self, self.loss)

        if log_freq is None:
            log_freq = max_epochs + 1

        if epoch_end_callback is None:
            epoch_end_callback = lambda _: False
        
        loss_history = []
        metrics_history = dict((m, []) for m in metrics)

        print("Training...")

        for ep in range(max_epochs):
            predictions = np.zeros((self.output_dim, num_inputs))

            for batch in self._random_batches(num_inputs, batch_size):
                batch_fwd = self.forward(inputs[:, batch])
                batch_pred, batch_inter = batch_fwd[-1], batch_fwd[:-1] 
                optimizer.optimize(targets[:, batch], batch_pred, batch_inter)

                predictions[:, batch] = batch_pred

            loss_history.append(np.mean(self.loss.evaluate(targets, predictions)))

            predictions = self.targets_to_labels(predictions)
            for m in metrics:
                metrics_history[m].append(metrics[m].evaluate(labels, predictions))

            if (ep + 1) % log_freq == 0:
                self._log(ep, max_epochs, loss_history, metrics_history)
            
            if epoch_end_callback(self):
                break

        self._log_final(ep, max_epochs, loss_history, metrics_history)

        return loss_history, metrics_history
    
    def _random_batches(
        self, num_inputs: int, batch_size: int | Literal["all"] = "all"
    ) -> Generator[nptype.NDArray, None, None]:
        if batch_size == "all":
            yield np.arange(num_inputs)
            return
        
        permutation = np.random.permutation(np.arange(num_inputs))
        yield from np.split(permutation, range(batch_size, num_inputs, batch_size))
    
    def _log(
        self, epoch: int, max_epochs: int,
        loss_history: List[float], metrics_history: Dict[str, List[float]]
    ):
        print(f"Epoch {epoch + 1:d}/{max_epochs}:",
                ", ".join([f"loss = {loss_history[-1]:.4f}"]
                          + [f"{m} = {metrics_history[m][-1]:.4f}"
                             for m in metrics_history]))
        
    def _log_final(
        self, epoch: int, max_epochs: int,
        loss_history: List[float], metrics_history: Dict[str, List[float]]
    ):
        print(f"Finished after {epoch + 1:d}/{max_epochs} epoch(s):",
                ", ".join([f"loss = {loss_history[-1]:.4f}"]
                          + [f"{m} = {metrics_history[m][-1]:.4f}"
                             for m in metrics_history]))
        print()
