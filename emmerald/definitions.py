from typing import Protocol


class SubgroupDescription(Protocol):
  def __eq__(self, __o: object) -> bool:
    ...


class Quality(Protocol):
  ...

