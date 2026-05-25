from typing import Any, Dict

import tensorflow as tf


class CosineDecayWithRestartsLearningRateSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    """Cosine decay with warm restarts and a fixed floor after ``total_steps``."""

    def __init__(
        self,
        initial_lr: float,
        min_lr: float,
        total_steps: int,
        first_decay_steps: int,
        t_mul: float = 2.0,
        m_mul: float = 1.0,
    ):
        super().__init__()
        if initial_lr <= 0:
            raise ValueError("initial_lr must be positive")
        if min_lr < 0:
            raise ValueError("min_lr must be non-negative")
        if min_lr > initial_lr:
            raise ValueError("min_lr must be <= initial_lr")
        if total_steps <= 0 or first_decay_steps <= 0:
            raise ValueError("total_steps and first_decay_steps must be positive")

        self.initial_lr = float(initial_lr)
        self.min_lr = float(min_lr)
        self.total_steps = int(total_steps)
        self.first_decay_steps = int(first_decay_steps)
        self.t_mul = float(t_mul)
        self.m_mul = float(m_mul)
        self.alpha = self.min_lr / self.initial_lr
        self._schedule = tf.keras.optimizers.schedules.CosineDecayRestarts(
            initial_learning_rate=self.initial_lr,
            first_decay_steps=self.first_decay_steps,
            t_mul=self.t_mul,
            m_mul=self.m_mul,
            alpha=self.alpha,
        )

    def __call__(self, step: tf.Tensor) -> tf.Tensor:
        step = tf.cast(step, tf.float32)
        lr = self._schedule(step)
        return tf.where(step < float(self.total_steps), lr, tf.cast(self.min_lr, lr.dtype))

    def get_config(self) -> Dict[str, Any]:
        return {
            "initial_lr": self.initial_lr,
            "min_lr": self.min_lr,
            "total_steps": self.total_steps,
            "first_decay_steps": self.first_decay_steps,
            "t_mul": self.t_mul,
            "m_mul": self.m_mul,
        }
