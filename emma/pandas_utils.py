from dataclasses import dataclass
from functools import reduce
import numpy as np
from typing import Any, Protocol, FrozenSet, List, Tuple


class LogicalOperator(Protocol):
    def __call__(self, df):
        ...

    def __repr__(self):
        ...


def bitwise_and(iterable):
    return reduce(np.bitwise_and, iterable, True)


def description_to_indices(X, description: List[LogicalOperator]):
    return X.loc[lambda d: bitwise_and(f(d) for f in description)].index


@dataclass(frozen=True)
class EqualsOperator:
    column: str
    value: Any

    def __call__(self, df):
        return df[self.column] == self.value

    def __repr__(self):
        return f'{self.column}=={self.value}'


@dataclass(frozen=True)
class NotEqualsOperator:
    column: str
    value: Any

    def __call__(self, df):
        return df[self.column] != self.value

    def __repr__(self):
        return f'{self.column}!={self.value}'


@dataclass(frozen=True)
class InSetOperator:
    column: str
    value: FrozenSet[Any]

    def __call__(self, df):
        return df[self.column].isin(self.value)

    def __repr__(self):
        return f'{self.column} in {self.value}'


@dataclass(frozen=True)
class InRangeOperator:
    """
    Range includes lower bound, and does not include upper bound
    """
    column: str
    range: Tuple[Any, Any]

    def __call__(self, df):
        return (df[self.column]>=self.range[0]) & (df[self.column]<self.range[1]) 

    def __repr__(self):
        return f'{self.column} in [{self.range[0]}, {self.range[1]})'
