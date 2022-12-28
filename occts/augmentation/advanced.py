"""_summary_
"""
import numpy as np
import occts.utils.dtw as dtw
from typing import List
from tqdm import tqdm
from .transform import Transform
from .basic import Jitter
from scipy.interpolate import CubicSpline


class MagnitudeWarp(Transform):
    """ The magnitude of each time series is multiplied by a curve
    created by cubicspline with a set of number of knots at random magnitudes.

    Args:
        sigma (float, optional): Standard deviation of the random magnitudes
            of the warping path. Defaults to 0.2.
        knot (int, optional): Number of hills/valleys. Defaults to 4.
    """

    def __init__(self, sigma: float = 0.2, knot: int = 4) -> None:
        super().__init__()
        self._sigma = sigma
        self._knot = knot

    def _transform(self, x: np.ndarray) -> np.ndarray:
        """ Apply the Magnitude Warping in a given time series.

        Args:
            x (np.ndarray): 3D numpy array-like time series in format
                (batch, time_steps, channel).

        Returns:
            np.ndarray: Numpy array-like of generated data in the same
                dimensions of the input.
        """
        original_steps = np.arange(x.shape[1])

        random_warps = np.random.normal(
            loc=1.0,
            scale=self._sigma,
            size=(x.shape[0], self._knot + 2, x.shape[0])
        )

        warp_steps = (
            np.ones((x.shape[2], 1)) *
            (np.linspace(0, x.shape[1] - 1., num=self._knot + 2))
        ).T

        ret = np.zeros_like(x)

        for i, pat in enumerate(x):
            warper = np.array([
                (
                    CubicSpline(warp_steps[:,dim],
                    random_warps[i,:,dim])(original_steps)
                ) for dim in range(x.shape[2])
            ]).T
            ret[i] = pat * warper

        return ret

    def __repr__(self) -> str:
        class_name = self.__class__.__name__
        return f'{class_name}(sigma={self._sigma}, knot={self._knot})'


class TimeWarp(Transform):
    """ Random smooth time warping.

    Args:
        sigma (float, optional): Standard deviation of the random magnitudes
            of the warping path. Defaults to 0.2.
        knot (int, optional): Number of hills/valleys on the warping path.
            Defaults to 4.
    """

    def __init__(self, sigma: float = 0.2, knot: int = 4) -> None:
        super().__init__()
        self._sigma = sigma
        self._knot = knot

    def _transform(self, x: np.ndarray) -> np.ndarray:
        """ Apply a Random Smooth in a given time series.

        Args:
            x (np.ndarray): 3D numpy array-like time series in format
                (batch, time_steps, channel).

        Returns:
            np.ndarray: Numpy array-like of generated data in the same
                dimensions of the input.
        """

        original_steps = np.arange(x.shape[1])

        random_warps = np.random.normal(
            loc=1.0,
            scale=self._sigma,
            size=(x.shape[0], self._knot + 2, x.shape[2])
        )

        warp_steps = (
            np.ones((x.shape[2], 1)) *
            (np.linspace(0, x.shape[1] - 1., num=self._knot + 2))
        ).T

        ret = np.zeros_like(x)

        for i, pat in enumerate(x):
            for dim in range(x.shape[2]):
                time_warp = CubicSpline(
                    warp_steps[:, dim],
                    warp_steps[:, dim] * random_warps[i, :, dim]
                )(original_steps)

                scale = (x.shape[1] - 1) / time_warp[-1]

                ret[i, :, dim] = np.interp(
                    original_steps,
                    np.clip(scale * time_warp, 0, x.shape[1] - 1),
                    pat[:, dim]
                ).T

        return ret

    def __repr__(self) -> str:
        class_name = self.__class__.__name__
        return f'{class_name}(sigma={self._sigma}, knot={self._knot})'


class WindowSlice(Transform):
    """ Cropping the time series by the reduce ratio.

    Args:
        reduce_ratio (float, optional): _description_. Defaults to 0.9.
    """

    def __init__(self, reduce_ratio: float = 0.9) -> None:
        super().__init__()
        self._reduce_ratio = reduce_ratio

    def _transform(self, x: np.ndarray) -> np.ndarray:
        """ Apply the window slice in a given time series.

        Args:
            x (np.ndarray): 3D numpy array-like time series in format
                (batch, time_steps, channel).

        Returns:
            np.ndarray: Numpy array-like of generated data in the same
                dimensions of the input.
        """
        target_len = np.ceil(self._reduce_ratio * x.shape[1]).astype(int)

        if target_len >= x.shape[1]:
            return x

        starts = np.random.randint(
            low=0,
            high=x.shape[1] - target_len,
            size=(x.shape[0])
        ).astype(int)
        ends = (target_len + starts).astype(int)

        ret = np.zeros_like(x)

        for i, pat in enumerate(x):
            for dim in range(x.shape[2]):
                ret[i, :, dim] = np.interp(
                    np.linspace(0, target_len, num=x.shape[1]),
                    np.arange(target_len),
                    pat[starts[i]:ends[i], dim]
                ).T

        return ret

    def __repr__(self) -> str:
        class_name = self.__class__.__name__
        return f'{class_name}(reduce_ratio={self._reduce_ratio})'


class WindowWarp(Transform):
    """ Randomly warps a window by scales.

    See:
        https://halshs.archives-ouvertes.fr/halshs-01357973/document

    Args:
        window_ratio (float, optional): Ratio of the window to the full time
            series. Defaults to 0.1.
        scales (List[float], optional): A list ratios to warp the window by.
            Defaults to [0.5, 2.0].
    """

    def __init__(
        self,
        window_ratio: float = 0.1,
        scales: List[float] = None
    ) -> None:
        super().__init__()
        self._window_ratio = window_ratio
        self._scales = scales if scales else [0.5, 2.0]

    def _transform(self, x: np.ndarray) -> np.ndarray:
        """ Apply Window Warp in a given time series.

        Args:
            x (np.ndarray): 3D numpy array-like time series in format
                (batch, time_steps, channel).

        Returns:
            np.ndarray: Numpy array-like of generated data in the same
                dimensions of the input.
        """
        warp_scales = np.random.choice(self._scales, x.shape[0])
        warp_size = np.ceil(self._window_ratio * x.shape[1]).astype(int)
        window_steps = np.arange(warp_size)

        window_starts = np.random.randint(
            low=1,
            high=x.shape[1]-warp_size-1,
            size=(x.shape[0])
        ).astype(int)
        window_ends = (window_starts + warp_size).astype(int)

        ret = np.zeros_like(x)

        for i, pat in enumerate(x):
            for dim in range(x.shape[2]):
                start_seg = pat[:window_starts[i],dim]
                window_seg = np.interp(
                    np.linspace(
                        0,
                        warp_size-1,
                        num=int(warp_size*warp_scales[i])
                    ),
                    window_steps, pat[window_starts[i]:window_ends[i],dim])
                end_seg = pat[window_ends[i]:,dim]

                warped = np.concatenate((start_seg, window_seg, end_seg))

                ret[i,:,dim] = np.interp(
                    np.arange(x.shape[1]),
                    np.linspace(0, x.shape[1] - 1., num=warped.size),
                    warped
                ).T

        return ret


    def __repr__(self) -> str:
        class_name = self.__class__.__name__
        return f'{class_name}(window_ratio={self._window_ratio},' \
            f' scales={self._scales})'


class Spawner(Transform):
    """ Uses SPAWNER to augment the time series.

    See:
        Based on: K. Kamycki, T. Kapuscinski, M. Oszust, "Data
            Augmentation with Suboptimal Warping for Time-Series
            Classification," Sensors, vol. 20, no. 1, 2020.

    Args:
        sigma (float, optional): Standard deviation of the jittering.
            Defaults to 0.05.
        verbose (int, optional): 1 prints out a DTW matrix.
            0 shows nothing. Defaults to 0.
    """

    def __init__(self, sigma: float = 0.05, verbose: int = 0) -> None:
        super().__init__()
        self._sigma = sigma
        self._verbose = verbose

    def _transform(self, x: np.ndarray, labels: np.ndarray) -> np.ndarray:
        """ Apply the SPAWNER in a given time series.

        Args:
            x (np.ndarray): 3D numpy array-like of time series in format
                (batch, time_steps, channel).
            labels (np.ndarray): 2D or 3D numpy array-like either list of
                integers or one hot of the labels.

        Returns:
            np.ndarray: Numpy array-like of generated data in the same
                dimensions of the input.
        """
        random_points = np.random.randint(
            low=1,
            high=x.shape[1] - 1,
            size=x.shape[0]
        )
        window = np.ceil(x.shape[1] / 10.).astype(int)

        original_steps = np.arange(x.shape[1])

        l = np.argmax(labels, axis=1) if labels.ndim > 1 else labels

        ret = np.zeros_like(x)

        for i, pat in enumerate(tqdm(x)):
            # guarantees that same one isnt selected
            choices = np.delete(np.arange(x.shape[0]), i)
            # remove ones of different classes
            choices = np.where(l[choices] == l[i])[0]

            if choices.size > 0:
                random_sample = x[np.random.choice(choices)]

                # SPAWNER splits the path into two randomly
                path_1 = dtw.dtw(
                    pat[: random_points[i]],
                    random_sample[:random_points[i]],
                    dtw.RETURN_PATH,
                    slope_constraint='symmetric',
                    window=window
                )

                path_2 = dtw.dtw(
                    pat[random_points[i]:],
                    random_sample[random_points[i]:],
                    dtw.RETURN_PATH,
                    slope_constraint='symmetric',
                    window=window,
                )

                combined = np.concatenate(
                    (np.vstack(path_1), np.vstack(path_2 + random_points[i])),
                    axis=1
                )

                mean = np.mean(
                    [pat[combined[0]], random_sample[combined[1]]],
                    axis=0
                )

                for dim in range(x.shape[2]):
                    ret[i, :, dim] = np.interp(
                        original_steps,
                        np.linspace(0, x.shape[1] - 1., num=mean.shape[0]),
                        mean[:, dim]
                    ).T
            else:
                if self._verbose > -1:
                    print('There is only one pattern of class %d, skipping pattern average' % l[i])
                ret[i, :] = pat

        return Jitter(sigma=self._sigma)(ret)

    def __call__(self, x: np.ndarray, labels: np.ndarray) -> np.ndarray:
        return self._transform(x, labels)

    def __repr__(self) -> str:
        class_name = self.__class__.__name__
        return f'{class_name}(sigma={self._sigma}, verbose={self._verbose})'


class WDBA(Transform):
    """ TODO: make the docstring

    Args:
        batch_size (int, optional): _description_. Defaults to 6.
        slope_constraint (str, optional): _description_. Defaults to symmetric.
        use_window (bool, optional): _description_. Defaults to True.
        verbose (int, optional): _description_. Defaults to 0.
    """

    def __init__(
        self,
        batch_size: int = 6,
        slope_constraint: str = 'symmetric',
        use_window: bool = True,
        verbose: int = 0
    ) -> None:
        super().__init__()
        self._batch_size = batch_size
        self._slope_constraint = slope_constraint
        self._use_window = use_window
        self._verbose = verbose

    def __call__(self, x: np.ndarray, labels: np.ndarray) -> np.ndarray:
        return self._transform(x, labels)

    def _transform(self, x: np.ndarray, labels: np.ndarray) -> np.ndarray:
        """_summary_

        Args:
            x (np.ndarray): _description_
            labels (np.ndarray): _description_

        Returns:
            np.ndarray: _description_
        """
        if self._use_window:
            window = np.ceil(x.shape[1] / 10.).astype(int)
        else:
            window = None

        l = np.argmax(labels, axis=1) if labels.ndim > 1 else labels

        ret = np.zeros_like(x)

        for i in tqdm(range(ret.shape[0])):
            choices = np.where(l == l[i])[0]
            if choices.size > 0:
                k = min(choices.size, self._batch_size)
                random_prototypes = x[np.random.choice(choices, k, replace=False)]

                dtw_matrix = np.zeros((k, k))
                for p, prototype in enumerate(random_prototypes):
                    for s, sample in enumerate(random_prototypes):
                        if p == s:
                            dtw_matrix[p, s] = 0.0
                        else:
                            dtw_matrix[p, s] = dtw.dtw(
                                prototype,
                                sample,
                                dtw.RETURN_VALUE,
                                slope_constraint=self._slope_constraint,
                                window=window
                            )

                medoid_id = np.argsort(np.sum(dtw_matrix, axis=1))[0]
                nearest_order = np.argsort(dtw_matrix[medoid_id])
                medoid_pattern = random_prototypes[medoid_id]

                average_pattern = np.zeros_like(medoid_pattern)
                weighted_sums = np.zeros((medoid_pattern.shape[0]))

                for nid in nearest_order:
                    if nid == medoid_id or dtw_matrix[medoid_id, nearest_order[1]] == 0.0:
                        average_pattern += medoid_pattern
                        weighted_sums += np.ones_like(weighted_sums)
                    else:
                        path = dtw.dtw(
                            medoid_pattern,
                            random_prototypes[nid],
                            dtw.RETURN_PATH,
                            slope_constraint=self._slope_constraint,
                            window=window
                        )
                        dtw_value = dtw_matrix[medoid_id, nid]
                        warped = random_prototypes[nid, path[1]]
                        weight = np.exp(
                            np.log(0.5) * dtw_value / dtw_matrix[medoid_id, nearest_order[1]]
                        )
                        average_pattern[path[0]] += weight * warped
                        weighted_sums[path[0]] += weight

                ret[i, :] = average_pattern / weighted_sums[:, np.newaxis]
            else:
                if self._verbose > -1:
                    print(
                        f'There is only one pattern of class {l[i]}' \
                            f', skipping pattern average'
                    )
                ret[i, :] = x[i]

        return ret

    def __repr__(self) -> str:
        class_name = self.__class__.__name__
        return f'{class_name}(batch_size={self._batch_size},' \
            f' {self._slope_constraint}, {self._use_window},' \
                f'{self._verbose})'


class RandomGuidedWarp(Transform):
    """_summary_

    Args:
        Transform (_type_): _description_
    """

    def __init__(
        self,
        slope_constraint: str = 'symmetric',
        use_window: bool = True,
        dtw_type: str = 'normal',
        verbose: int = 0
    ) -> None:
        super().__init__()
        self._slope_constraint = slope_constraint
        self._use_window = use_window
        self._dtw_type = dtw_type
        self._verbose = verbose

    def __call__(self, x: np.ndarray, labels: np.ndarray) -> np.ndarray:
        return self._transform(x, labels)

    def _transform(self, x: np.ndarray, labels: np.ndarray) -> np.ndarray:
        if self._use_window:
            window = np.ceil(x.shape[1] / 10.0).astype(int)
        else:
            window = None

        original_step = np.arange(x.shape[1])
        l = np.argmax(labels, axis=1) if labels.ndim > 1 else labels

        ret = np.zeros_like(x)

        for i, pat in enumerate(tqdm(x)):
            choices = np.delete(np.arange(x.shape[0]), i)
            choices = np.where(l[choices] == l[i])[0]
            if choices.size > 0:
                random_prototype = x[np.random.choice(choices)]
                if self._dtw_type == 'shape':
                    raise ValueError()
                else:
                    path = dtw.dtw(
                        random_prototype,
                        pat,
                        dtw.RETURN_PATH,
                        slope_constraint=self._slope_constraint,
                        window=window
                    )

                warped = pat[path[1]]
                for dim in range(x.shape[2]):
                    ret[i, :, dim] = np.interp(
                        original_step,
                        np.linspace(0, x.shape[1] - 1, num=warped.shape[0]),
                        warped[:, dim]
                    ).T
            else:
                if self._verbose > -1:
                    print(
                        f'There is only one pattern of class {l[i]}' \
                            ', skipping time warping'
                    )
                ret[i, :] = pat

        return ret

    def __repr__(self) -> str:
        class_name = self.__class__.__name__
        return f'{class_name}(slope_constraint={self._slope_constraint},' \
            f' use_window{self._use_window}, dtw_type={self._dtw_type},' \
                f' verbose={self._verbose})'
