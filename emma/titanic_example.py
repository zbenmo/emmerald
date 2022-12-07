from sklearn.datasets import fetch_openml
from EMM import EMM
import numpy as np
from typing import Any, Protocol, FrozenSet, List
from dataclasses import dataclass
# np.random.seed(42)
from functools import reduce


def titanic_example():
  X, y = fetch_openml("titanic", version=1, as_frame=True, return_X_y=True)
  X.drop(['boat', 'body', 'home.dest'], axis=1, inplace=True)
  y = y.astype(int)
  X.pclass = X.pclass.astype(int)

  # print(X.info())
  # #   Column    Non-Null Count  Dtype
  # ---  ------    --------------  -----
  # 0   pclass    1309 non-null   float64
  # 1   name      1309 non-null   object
  # 2   sex       1309 non-null   category
  # 3   age       1046 non-null   float64
  # 4   sibsp     1309 non-null   float64
  # 5   parch     1309 non-null   float64
  # 6   ticket    1309 non-null   object
  # 7   fare      1308 non-null   float64
  # 8   cabin     295 non-null    object
  # 9   embarked  1307 non-null   category

  print(X.pclass.unique())
  print(X.sex.unique())
  print(X.embarked.unique())

  class LogicalOperator(Protocol):
    def __call__(self, df):
      ...

    def __repr__(self):
      ...

  def bitwise_and(iterable):
      return reduce(np.bitwise_and, iterable, True)

  def description_to_indices(description: List[LogicalOperator]):
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

  description_options = {
    'pclass': [
      EqualsOperator('pclass', pclass) for pclass in X.pclass.unique()
    ] + [
      InSetOperator('pclass', pclassSet) for pclassSet in [{1, 2}, {2, 3}]
    ],
    'sex': [
      EqualsOperator('sex', sex) for sex in X.sex.unique()
    ],
    'embarked': [
      EqualsOperator('embarked', port) for port in X.embarked.unique() # TODO: check if for None it actually works
    ] + [
      NotEqualsOperator('embarked', port) for port in X.embarked.unique()
    ]
  }

  def quality(description):
    indices = description_to_indices(description)
    return (np.mean(y[indices]), sum(indices), -len(description))

  def refinment(description):
    for _, options in description_options.items():
      for option in options:
        if description and (option in description):
          continue
        y = [] if description is None else description[:]
        y.append(option)
        yield sorted(y, key=str) # we sort the description here so that it will be easier to catch duplicates

  def satisfies(description):
    indices = description_to_indices(description)
    return sum(indices) > 10

  emm = EMM(
      dataset=None,
      quality_func=quality,
      refinment_func=refinment,
      satisfies_all_func=satisfies
      )

  results = emm.most_exceptional()

  for result in results:
    print(result)


if __name__ == "__main__":
    titanic_example()