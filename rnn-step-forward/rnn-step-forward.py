import numpy as np

def rnn_step_forward(x_t: list, h_prev: list, Wx: list, Wh: list, b: list) -> np.ndarray:
    """
    Returns a NumPy array with shape (H,).
    """
    ford=np.dot(x_t,Wx) + np.dot(h_prev,Wh) + b
    ht=np.tanh(ford)
    return ht