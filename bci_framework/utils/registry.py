"""Generic registry for pluggable components (preprocessing, features, classifiers)."""

from typing import Callable, TypeVar

T = TypeVar("T")


class Registry:
    """Registry that maps string names to classes or callables for easy extension."""

    def __init__(self, name: str = "registry") -> None:
        self._name = name
        self._store: dict[str, type | Callable[..., T]] = {}

    def register(self, name: str | None = None) -> Callable[[type | Callable[..., T]], type | Callable[..., T]]:
        """Decorator to register a class or callable."""

        def decorator(obj: type | Callable[..., T]) -> type | Callable[..., T]:
            key = name if name is not None else getattr(obj, "__name__", str(obj))
            self._store[key] = obj
            return obj

        return decorator

    def get(self, name: str) -> type | Callable[..., T]:
        """Get registered component by name."""
        if name not in self._store:
            raise KeyError(f"Unknown {self._name} '{name}'. Available: {list(self._store.keys())}")
        return self._store[name]

    def list_names(self) -> list[str]:
        """List all registered names."""
        return list(self._store.keys())

    def __contains__(self, name: str) -> bool:
        return name in self._store
