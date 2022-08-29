""" Module containing the base structure for a transformer.

Concepts:
    AbstractFactory
    Composite
"""
import abc
import numpy as np
from abc import abstractmethod
from typing import Any, List


class Transform(abc.ABC):
    """ Base transform class, works as an abstract class to implement
    other transforms on top of it.

    Based on Abstract Factory design pattern.
    """

    def __init__(self) -> None:
        pass

    @abstractmethod
    def _transform(self, x: Any) -> Any:
        pass

    def __call__(self, x: Any) -> Any:
        return self._transform(x)

    def __repr__(self) -> str:
        class_name = self.__class__.__name__
        return f'{class_name}()'


class Compose:
    """ Compose is like a iterator of Transforms. A compose receive a list of
    Transforms and iterate over them applying each transform in the input and
    returning them after the end.
    """

    def __init__(self, transforms: List[Transform]) -> None:
        self._transforms = transforms

    def __call__(self, x: Any) -> Any:
        for transform in self._transforms:
            x = transform(x)
        return x

    def __repr__(self) -> str:
        class_name = self.__class__.__name__
        transform_names = [
            t.__repr__() for t in self._transforms
        ]

        return '{}(\n{}\n)'.format(class_name, ',\n'.join(transform_names))


class DummyTransform(Transform):
    """ A dummy transform that only returns the actual input as an output.
    """

    def _transform(self, x: np.ndarray) -> np.ndarray:
        return x

    def __repr__(self) -> str:
        class_name = self.__class__.__name__
        return '{}()'.format(class_name)
