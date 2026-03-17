import heapq
from dataclasses import dataclass
from typing import TypeVar

T = TypeVar("T")


@dataclass(order=True)
class PrioritizedItem:
    priority: float
    sequence: int
    data: object


class PriorityQueue:
    def __init__(self):
        self._heap: list[PrioritizedItem] = []
        self._counter = 0

    def push(self, priority: float, data: object) -> None:
        item = PrioritizedItem(priority, self._counter, data)
        self._counter += 1
        heapq.heappush(self._heap, item)

    def pop(self) -> object:
        item = heapq.heappop(self._heap)
        return item.data

    def top_k(self, k: int) -> list[object]:
        return [heapq.heappop(self._heap).data for _ in range(min(k, len(self._heap)))]


def sort_with_uuid_tiebreaker(items: list[dict]) -> list[dict]:
    return sorted(items, key=lambda x: (x["value"], x["uuid"]))
