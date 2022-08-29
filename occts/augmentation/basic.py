""" _summary_
"""
import numpy as np
from .transform import Transform


class Jitter(Transform):
    """ Apply jittering or noise to the time series.

    See:
        https://arxiv.org/pdf/1706.00527.pdf

    Args:
        sigma (float, optional): Standard deviation of the distribution
            to be added. Defauts to 0.03.
    """

    def __init__(self, sigma: float = 0.03) -> None:
        super().__init__()
        self._sigma = sigma

    def _transform(self, x: np.ndarray) -> np.ndarray:
        """ Apply the Jitter in a input x.

        Args:
            x (np.ndarray): 3D numpy array-like of time series in format
                (batches, time_steps, channels).

        Returns:
            np.ndarray: Numpy array-like of generated data in the same
                dimensions of the input.
        """
        return x + np.random.normal(loc=0, scale=self._sigma, size=x.shape)

    def __repr__(self) -> str:
        class_name = self.__class__.__name__
        return f'{class_name}(sigma={self._sigma})'


class Scale(Transform):
    """ Scale each time series by a constant ammount.

    See:
        https://arxiv.org/pdf/1706.00527.pdf

    Args:
        sigma (float, optional): Standard deviation of the scaling
            constant. Defaults to 0.1.
    """

    def __init__(self, sigma: float = 0.1) -> None:
        super().__init__()
        self._sigma = sigma

    def _transform(self, x: np.ndarray) -> np.ndarray:
        """ Scale a time series by a constant ammount.

        Args:
            x (np.ndarray): 3D numpy array-like of time series in format
                (batch, time_steps, channels).

        Returns:
            np.ndarray: Numpy array-like of generated data in the same
                dimensions of the input.
        """
        factor = np.random.normal(
            loc=1,
            scale=self._sigma,
            size=(x.shape[0], x.shape[2])
        )
        return np.multiply(x, factor[:, np.newaxis, :])

    def __repr__(self) -> str:
        class_name = self.__class__.__name__
        return f'{class_name}(sigma={self._sigma})'


class Rotate(Transform):
    """ Perform a Rotate transform in a batch of time series.
    """

    def _transform(self, x: np.ndarray) -> np.ndarray:
        """ For 1D time series, randomly flipping. For multivariate time
        series flipping as well as axis shuffling.

        Args:
            x (np.ndarray): 3D numpy array-like of time series in format
                (batch, time_steps, channels).

        Returns:
            np.ndarray: Numpy array-like of generated data in the same
                dimensions of the input.
        """
        flip = np.random.choice([-1, 1], size=(x.shape[0],x.shape[2]))
        rotate_axis = np.arange(x.shape[2])
        np.random.shuffle(rotate_axis)
        return flip[:,np.newaxis,:] * x[:,:,rotate_axis]

    def __repr__(self) -> str:
        class_name = self.__class__.__name__
        return f'{class_name}()'


class Permute(Transform):
    """ Applies a permutation in the time series batch.

    Args:
        max_segments(int, optional): The maximum number of segments to use.
            The minimum number is 1. Defaults to 5.
        seg_mode (str, optional): Equal uses equal sized segments and random
            uses randomly sized segments. Defaults to equal.
    """

    def __init__(self, max_segments: int = 5, seg_mode: str = 'equal') -> None:
        super().__init__()

        assert max_segments >= 1, 'The minimum number of max_segments is 1.'

        self._max_segments = max_segments
        self._seg_mode = seg_mode

    def _transform(self, x: np.ndarray) -> np.ndarray:
        """ Random permutation of segments. A random number of segments is
            used, up to max_segments.

        Args:
            x (np.ndarray): 3D numpy array-like time series in format
                (batches, time_steps, channels).

        Returns:
            np.ndarray: Numpy array-like of generated data in the same
                dimensions of the input.
        """
        orig_steps = np.arange(x.shape[1])

        num_segs = np.random.randint(1, self._max_segments, size=(x.shape[0]))

        ret = np.zeros_like(x)
        for i, pat in enumerate(x):
            if num_segs[i] > 1:
                if self._seg_mode == 'random':
                    split_points = np.random.choice(
                        x.shape[1] - 2,
                        num_segs[i] - 1,
                        replace=False
                    )
                    split_points.sort()
                    splits = np.split(orig_steps, split_points)
                else:
                    splits = np.array_split(orig_steps, num_segs[i])
                warp = np.concatenate(np.random.permutation(splits)).ravel()
                ret[i] = pat[warp]
            else:
                ret[i] = pat
        return ret
