import numpy as np

from lahja.metrics import expected_calibration_error, fit_temperature, softmax


def test_ece_perfect_and_overconfident():
    y = np.array([0, 1, 0, 1])
    perfect = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]])
    assert expected_calibration_error(perfect, y) == 0.0
    # always 90 % sure, right half the time: ECE = |0.5 - 0.9| = 0.4
    over = np.array([[0.9, 0.1], [0.9, 0.1], [0.9, 0.1], [0.9, 0.1]])
    assert abs(expected_calibration_error(over, y) - 0.4) < 1e-9


def test_temperature_recovers_scale():
    rng = np.random.default_rng(0)
    true_logits = rng.normal(size=(4000, 5)) * 2
    y = np.array([rng.choice(5, p=p) for p in softmax(true_logits)])
    T = fit_temperature(true_logits * 3, y)  # the model is 3x overconfident
    assert abs(T - 3) < 0.3
    assert (softmax(true_logits * 3 / T).argmax(1) == (true_logits * 3).argmax(1)).all()
