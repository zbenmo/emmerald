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
import pandas as pd 
from collections import defaultdict
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder, LabelBinarizer
from sklearn.pipeline import make_pipeline
from sklearn.compose import make_column_transformer


def counted(f):
    def wrapped(*args, **kwargs):
        wrapped.calls += 1
        return f(*args, **kwargs)
    wrapped.calls = 0
    return wrapped


def adult_example():
  """
  In this example we build for every subgroup a decision tree of max depth 3 and calculate the roc_auc_score (on the training set).
  We then look for the groups that are hardest to nail (with decision tree of max depth 3).
  """

  X, y = fetch_openml("adult", version=1, as_frame=True, return_X_y=True)
  # print(X.info())
  #  #   Column          Non-Null Count  Dtype
  # ---  ------          --------------  -----
  #  0   age             48842 non-null  category
  #  1   workclass       46043 non-null  category
  #  2   fnlwgt          48842 non-null  float64
  #  3   education       48842 non-null  category
  #  4   education-num   48842 non-null  float64
  #  5   marital-status  48842 non-null  category
  #  6   occupation      46033 non-null  category
  #  7   relationship    48842 non-null  category
  #  8   race            48842 non-null  category
  #  9   sex             48842 non-null  category
  #  10  capitalgain     48842 non-null  category
  #  11  capitalloss     48842 non-null  category
  #  12  hoursperweek    48842 non-null  category
  #  13  native-country  47985 non-null  category

  # print(y.unique())
  # ['<=50K', '>50K']

  y = y.map({'<=50K': 0, '>50K': 1}).astype(int)

  description_options = defaultdict(list)

  for column in X.select_dtypes(include=['category']).columns:
    description_options[column].extend(
      EqualsOperator(column, val) for val in X[column].unique()
    )

  # If makes sense, we can preprocess the full dataset as to save per-subgroup work later.
  # Make sure such an optimization, if indeed time gain, does not hinder what you are exploring (train/test split considerations etc.) 
  dataset_features = pd.get_dummies(X)

  @counted
  def quality(description):
    indices = description_to_indices(X, description)
    # features = pd.get_dummies(X.loc[indices])
    features = dataset_features.loc[indices]
    targets = y.loc[indices]
    dt = DecisionTreeClassifier(max_depth=3)
    dt.fit(features, targets)
    predictions = dt.predict_proba(features)
    roc_auc = roc_auc_score(targets, predictions[:, 1])
    size_of_subgroup = len(indices)
    return (-roc_auc, size_of_subgroup, -len(description))

  refinment = make_refinment(None, description_options)
  refinment = counted(refinment)

  @counted
  def satisfies(description):
    indices = description_to_indices(X, description)
    avg = np.mean(y.loc[indices])
    return len(indices) > 100 and (0.1 < avg < 0.9) # so at least 101 in the subgroup and some variablity in the target

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
    adult_example()