from sklearn.datasets import fetch_openml
from EMM import EMM
import numpy as np
# np.random.seed(42)


def titanic_example():
  X, y = fetch_openml("titanic", version=1, as_frame=True, return_X_y=True)
  y = y.astype(int)
  X.drop(['boat', 'body', 'home.dest'], axis=1, inplace=True)
  # X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2)
  # pass

  def description_to_indices(description):
    query = ' and '.join(map(lambda x: f'({x[0]}{x[1]}{x[2]})', description))
    return X.query(query).index

  def quality(description):
    indices = description_to_indices(description)
    return np.mean(y[indices])

  def refinment(description):
    for column in X.columns:
      y = [] if description is None else description[:]
      y.append((column, '==', 3))
      yield y

  def satisfies(description):
    indices = description_to_indices(description)
    return sum(indices) > 10

  emm = EMM(
      dataset=None,
      quality_func=quality,
      refinment_func=refinment,
      satisfies_all_func=satisfies
      )

  result_set = emm.most_exceptional()

  while not result_set.empty():
    print(result_set.pop())


if __name__ == "__main__":
    titanic_example()