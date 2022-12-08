from typing import Protocol, Generator

from .definitions import *
from .utils import *


class QualityFunc(Protocol):
  def __call__(description: SubgroupDescription) -> Quality:
    ...


class RefinmentFunc(Protocol):
  def __call__(description: SubgroupDescription) -> Generator[SubgroupDescription, None, None]:
    ...


class SatisfiesAllFunc(Protocol):
  def __call__(description: SubgroupDescription) -> bool:
    ...


class EMM:
  """
  This Exceptional Model Mining is a generic algorithm to find interesting subgroups with some respect.
  Subgroups are those that also have an interpratable description. 
  """

  def __init__(
      self,
      dataset: SubgroupDescription,
      quality_func: QualityFunc,
      refinment_func: RefinmentFunc,
      satisfies_all_func: SatisfiesAllFunc
    ):
    """
    dataset is the description of the whole dataset (ex. None, "All", etc.)
    quality_func is a callable that returns a quality given a description
    refinment_func is a callable that yields all possible refinments of a given seed description to an additional one level
    satisfies_all_func is a constraint that each resulting description must meet. It is also a way to bound the search.
      It is assumed that once a description does not meet a constraint, neither does any refinment of that description.  
    """

    self.dataset = dataset
    self.quality_func = quality_func
    self.refinment_func = refinment_func
    self.satisfies_all_func = satisfies_all_func

  def most_exceptional(self,
          levels_deep: int = 3, width_of_search: int = 10, top_q: int = 10):
    """
    This is a Beam-Search starting with short descriptions, which are the first level,
    and then refining the descriptions level by level to a maximum of 'levels_deep'.
    """

    candidates_queue = SimpleQueue()
    candidates_queue.push(self.dataset)
    results_set = PriorityQueue(top_q)
    for _ in range(levels_deep):
      beam = PriorityQueue(width_of_search)
      while not candidates_queue.empty():
        seed_description = candidates_queue.pop()
        for description in self.refinment_func(seed_description):
          if not self.satisfies_all_func(description):
            continue
          quality = self.quality_func(description)
          element = PriorityQueueElement(quality, description)
          results_set.push(element)
          beam.push(element)
      while not beam.empty():
        item_from_beam = beam.pop()
        candidates_queue.push(item_from_beam.description)
    results = []
    while not results_set.empty():
      results.insert(0, results_set.pop())
    return results
