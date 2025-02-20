from abc import ABC
import app_config as cf
import baseline_config as bl_cf


class _ConfigMixin(ABC):
    def has_cf(self):
        return cf.is_available()

    def has_bl_cf(self):
        return bl_cf.is_available()

    def get_cf(self):
        return cf.get_config()

    def get_bl_cf(self):
        return bl_cf.get_bl_config()
