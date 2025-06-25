from abc import ABC, abstractmethod


class WithUpdateFromDictMixin(ABC):
    def __init__(self, state: dict):
        self.update_from_dict(state)

    def patch_state(self, state: dict):
        self.update_from_dict(state)
        for key, value in state.items():
            if hasattr(self, key):
                attr = getattr(self, key)
                if isinstance(attr, WithUpdateFromDictMixin) and isinstance(value, dict):
                    attr.patch_state(value)

    @abstractmethod
    def update_from_dict(self, state: dict) -> None:
        pass

    def _getattr(self, attr: str):
        try:
            return getattr(self, attr)
        except:
            return None
