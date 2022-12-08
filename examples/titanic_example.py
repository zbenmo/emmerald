from sklearn.datasets import fetch_openml
from emma import EMM
from emma.pandas_utils import (
  EqualsOperator,
  NotEqualsOperator,
  InSetOperator,
  InRangeOperator,
  description_to_indices,
  make_refinment
)
import numpy as np


def counted(f):
    def wrapped(*args, **kwargs):
        wrapped.calls += 1
        return f(*args, **kwargs)
    wrapped.calls = 0
    return wrapped


def titanic_example():
  """
  This is a basic example in which we look for subgroups where the survival rate is high.
  This is more a "subgroup discovery" task rather then full fledged EMM, as we don't build a model but just look at the target 'survived'.
  Still this example provides most of the elements for almost every EMM model class. 
  """

  X, y = fetch_openml("titanic", version=1, as_frame=True, return_X_y=True)
  X.drop(['boat', 'body', 'home.dest'], axis=1, inplace=True)
  y = y.astype(int)
  X.pclass = X.pclass.astype(int)
  X.sibsp = X.sibsp.astype(int)
  X.parch = X.parch.astype(int)

  # print(X.info())
  #  #   Column    Non-Null Count  Dtype
  # ---  ------    --------------  -----
  #  0   pclass    1309 non-null   int64
  #  1   name      1309 non-null   object
  #  2   sex       1309 non-null   category
  #  3   age       1046 non-null   float64
  #  4   sibsp     1309 non-null   int64
  #  5   parch     1309 non-null   int64
  #  6   ticket    1309 non-null   object
  #  7   fare      1308 non-null   float64
  #  8   cabin     295 non-null    object
  #  9   embarked  1307 non-null   category

  age_brackets = [round(x) for x in np.linspace(0, X.age.max() + 1, 8)]

  description_options = {
    'pclass': [
      EqualsOperator('pclass', pclass) for pclass in X.pclass.unique()
    ] + [
      InSetOperator('pclass', pclassSet) for pclassSet in [{1, 2}, {2, 3}]
    ],
    'sex': [
      EqualsOperator('sex', sex) for sex in X.sex.unique()
    ],
    'age': [
      InRangeOperator('age', (range_low, range_high)) for range_low, range_high in zip(age_brackets[:-1], age_brackets[1:])
    ],
    'embarked': [
      EqualsOperator('embarked', port) for port in X.embarked.unique()
    ] + [
      NotEqualsOperator('embarked', port) for port in X.embarked.unique()
    ]
  }

  @counted
  def quality(description):
    indices = description_to_indices(X, description)
    mean_survived = np.mean(y[indices])
    size_of_subgroup = len(indices)
    return (mean_survived, size_of_subgroup, -len(description))

  refinment = make_refinment(None, description_options)
  refinment = counted(refinment)

  @counted
  def satisfies(description):
    indices = description_to_indices(X, description)
    return len(indices) > 10 # so at least 11 in the subgroup

  emm = EMM(
      dataset=None,
      quality_func=quality,
      refinment_func=refinment,
      satisfies_all_func=satisfies
      )

  results = emm.most_exceptional(top_q=15)

  for result in results:
    print(result)

  print()
  print(f'{quality.calls=}')
  print(f'{refinment.calls=}')
  print(f'{satisfies.calls=}')


if __name__ == "__main__":
    titanic_example()