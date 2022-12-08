from dataclasses import dataclass, field
import heapq

from .definitions import *


class SimpleQueue:
  """
  FIFO and not bounded, yet does not add an item that is already in the queue.
  Not as efficient as can be, given that it scans the existing item on every push.
  """
  def __init__(self):
    self.queue = []

  def push(self, element: SubgroupDescription):
    """
    Adds to the tail of the queue yet only if not already in the queue.
    """
    if element in self.queue:
      return
    self.queue.append(element)

  def pop(self) -> SubgroupDescription:
    return self.queue.pop(0)

  def empty(self) -> bool:
    return len(self.queue) < 1


@dataclass(order=True)
class PriorityQueueElement:
  quality: Quality
  description: SubgroupDescription = field(compare=False)


class PriorityQueue:
  """
  Maintains at most max_items. Smaller items leaves first.
  Does not add an item that is already in the queue.
  Not as efficient as can be, given that it scans the existing item on every push.
  """
  def __init__(self, max_items: int):
    self.max_items = max_items
    self.heap = []

  def push(self, element: SubgroupDescription):
    """
    Adds to the heap, maintaining heap property, yet only if not already in the heap.
    """
    if element in self.heap:
      return
    if len(self.heap) < self.max_items:
      heapq.heappush(self.heap, element)
    else:
      _ = heapq.heappushpop(self.heap, element)

  def pop(self) -> SubgroupDescription:
    """
    With current implementation it pops and returns the smallest item.
    """
    return heapq.heappop(self.heap)

  def empty(self) -> bool:
    return len(self.heap) < 1
