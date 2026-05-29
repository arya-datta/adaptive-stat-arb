r"""Stage 8: deep-learning extensions (optional capstone, done carefully).

The goal is to *enhance* the principled model, not replace it with a black
box. We explicitly avoid the weak "LSTM predicts the spread" version. The one
principled, dependency-free component implemented here is a **Neural SDE**:

.. math:: dX_t = f_\theta(X_t)\,dt + \sigma\,dW_t,

where the drift :math:`f_\theta` is a small neural network (NumPy MLP, manual
backprop + Adam) — a data-driven *generalisation* of the OU drift
:math:`\kappa(\mu - X)`, retaining the SDE structure rather than discarding it.

**Gate (roadmap):** the neural component must beat its classical counterpart
under the *same* honest out-of-sample comparison, or it stays out — and saying
so is itself a sign of maturity. :func:`neural_vs_ou_gate` runs exactly that
comparison; on genuine OU data the linear drift wins (the neural net earns
nothing), while on a nonlinear (double-well) drift the neural SDE wins.

Reference: Neural SDE literature (Li et al. 2020; Kidger et al.), kept
subordinate to the stochastic-process core.
"""

from .neural_sde import NeuralSDE, MLP, neural_vs_ou_gate

__all__ = ["NeuralSDE", "MLP", "neural_vs_ou_gate"]
