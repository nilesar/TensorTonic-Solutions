import numpy as np

def dropout(
    x: list,
    p: float = 0.5,
    rng: np.random.Generator = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns (output, dropout_pattern) as NumPy arrays matching the shape of x.
    """
    x_arr=np.array(x)
    if rng is None:
        r=np.random.random(x_arr.shape)
    else:
        r=rng.random(x_arr.shape)


    keep_prob=1-p
    mask= r< keep_prob
    scale = 1.0 / keep_prob
    dropout_pattern = mask * scale
    output=x_arr*dropout_pattern
    return output,dropout_pattern
        