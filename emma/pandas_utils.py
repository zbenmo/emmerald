from dataclasses import dataclass
from functools import reduce
import numpy as np
from typing import Any, Protocol, FrozenSet, List, Tuple, Generator
from .EMM import RefinmentFunc, SubgroupDescription


class LogicalOperator(Protocol):
    def __call__(self, df):
        ...

    def __repr__(self):
        ...


def bitwise_and(iterable):
    return reduce(np.bitwise_and, iterable, True)


def description_to_indices(df, description: List[LogicalOperator]):
    return df.loc[lambda d: bitwise_and(f(d) for f in description)].index


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


def make_refinment(dataset: SubgroupDescription, description_options: dict) -> RefinmentFunc:
  """
  dataset is the representation of the whole dataset (ex. None, or 'ALL', etc.)
  description_options is a dictionality from a column to a list of relevant LogicalOperators available for a refinment
  """

  def refinment(description: List[LogicalOperator]) -> Generator[SubgroupDescription, None, None]:
    for _, options in description_options.items():
      for option in options:
        if description == dataset:
          refined_description = [option]
        else:
          if option in description:
            continue # just skip this option, as it is redundant
          refined_description = description[:]
          should_skip = False
          for desc in description:
            if desc.column != option.column:
              continue
            if isinstance(desc, EqualsOperator) and isinstance(option, EqualsOperator):
               should_skip = True
               break
            elif isinstance(desc, EqualsOperator) and isinstance(option, InSetOperator):
               should_skip = True
               break
            elif isinstance(desc, InSetOperator) and isinstance(option, EqualsOperator):
              if option.value in desc.value:
                refined_description.remove(desc)
            elif isinstance(desc, InSetOperator) and isinstance(option, InSetOperator):
                new_set = desc.value & option.value
                if len(new_set) < 1:
                  should_skip = True
                  break
                refined_description.remove(desc)
                if len(new_set) == 1:
                  option = EqualsOperator(option.column, new_set.pop())
                else:
                  option = InSetOperator(option.column, new_set)
          if should_skip:
            continue
          refined_description.append(option)
        yield sorted(refined_description, key=str) # we sort the refined_description here so that it will be easier to catch duplicates
  return refinment
