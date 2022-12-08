import pytest

from emmnesia.utils import (
    SimpleQueue,
    PriorityQueue,
    PriorityQueueElement
)


def test_simple_queue():
    queue = SimpleQueue()

    for item in [4, 1, 6, 7, 6]:
        queue.push(item)

    assert not queue.empty()
    assert queue.pop() == 4
    assert queue.pop() == 1
    assert queue.pop() == 6
    assert queue.pop() == 7
    assert queue.empty() # because when pushing 6 again it should have not been added


def test_priority_queue():
    queue = PriorityQueue(3)

    for item in [(4, 'a'), (1, 'b'), (6, 'c'), (7, 'd')]:
        queue.push(PriorityQueueElement(*item))

    assert not queue.empty()
    assert queue.pop() == PriorityQueueElement(4, 'a')
    assert queue.pop() == PriorityQueueElement(6, 'a')
    assert queue.pop() == PriorityQueueElement(7, 'a')
    assert queue.empty()


def test_priority_queue_adding_existing():
    queue = PriorityQueue(3)

    for item in [(4, 'a'), (1, 'b'), (6, 'c'), (7, 'd'), (7, 'd'), (7, 'd'), (7, 'd')]:
        queue.push(PriorityQueueElement(*item))

    assert not queue.empty()
    assert queue.pop() == PriorityQueueElement(4, 'a')  # because when pushing 7 again it should have not been added
    assert queue.pop() == PriorityQueueElement(6, 'a')
    assert queue.pop() == PriorityQueueElement(7, 'a')
    assert queue.empty() 
