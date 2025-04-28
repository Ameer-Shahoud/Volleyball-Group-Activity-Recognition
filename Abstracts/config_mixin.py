from abc import ABC
import sys
import app_config as cf
import baseline_config as bl_cf
from IPython.display import clear_output


class _ConfigMixin(ABC):
    """
    Mixin class to provide access to configuration settings.

    Methods:
        has_cf(): Checks if application configuration is available.
        has_bl_cf(): Checks if baseline configuration is available.
        get_cf(): Retrieves application configuration.
        get_bl_cf(): Retrieves baseline configuration.
    """

    def has_cf(self):
        return cf.is_available()

    def has_bl_cf(self):
        return bl_cf.is_available()

    def get_cf(self):
        return cf.get_config()

    def get_bl_cf(self):
        return bl_cf.get_bl_config()

    def clear_output(self):
        if cf.get_config().is_notebook:
            clear_output(wait=True)
        else:
            sys.stdout.write("\033c")
            sys.stdout.flush()
