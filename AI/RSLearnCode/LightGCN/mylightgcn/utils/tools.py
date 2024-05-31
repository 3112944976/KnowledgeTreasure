import numpy as np


def minibatch(*tensors, batch_size):
    """
    Create minibatches from input tensors.
    Args: tensors (tuple): Tuple of input tensors.
    Yields: tuple: Tuple of minibatches.
    """
    if len(tensors) == 1:
        tensor = tensors[0]
        for i in range(0, len(tensor), batch_size):
            yield tensor[i: i + batch_size]
    else:
        for i in range(0, len(tensors[0]), batch_size):
            yield tuple(x[i: i + batch_size] for x in tensors)


def shuffle(*arrays, indices=False):
    """
    Shuffle input arrays.
    Args:
        arrays (tuple): Tuple of input arrays.
        indices (bool, optional): Whether to return the shuffled indices.
                                  Defaults to False.
    Returns:
        tuple or np.array: Shuffled arrays or tuple of shuffled arrays and
                           shuffled indices.
    """
    if len(set(len(x) for x in arrays)) != 1:
        raise ValueError("All inputs to shuffle must have the same length.")

    shuffle_indices = np.arange(len(arrays[0]))
    np.random.shuffle(shuffle_indices)

    if len(arrays) == 1:
        result = arrays[0][shuffle_indices]
    else:
        result = tuple(x[shuffle_indices] for x in arrays)

    if indices:
        return result, shuffle_indices
    else:
        return result
