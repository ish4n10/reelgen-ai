from __future__ import annotations


class ObjectRegistry:
    def __init__(self) -> None:
        self._ids_by_object_id: dict[int, str] = {}
        self._next_id = 1

    def get_id(self, mobject) -> str:
        object_id = id(mobject)
        explicit_id = getattr(mobject, "_reelaigen_id", None)
        if explicit_id:
            self._ids_by_object_id[object_id] = str(explicit_id)
            return str(explicit_id)
        if object_id not in self._ids_by_object_id:
            self._ids_by_object_id[object_id] = f"obj_{self._next_id:04d}"
            self._next_id += 1
        return self._ids_by_object_id[object_id]

    def forget(self, mobject) -> None:
        self._ids_by_object_id.pop(id(mobject), None)

    def clear(self) -> None:
        self._ids_by_object_id.clear()
        self._next_id = 1
