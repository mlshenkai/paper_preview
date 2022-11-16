# -*- coding: utf-8 -*-
# @Author: watcher
# @Created Time: 2022/11/11 3:36 PM
# @File: deprecation_utils
# @Email: mlshenkai@163.com
import warnings
from functools import wraps
from inspect import Parameter, signature
from typing import Optional, Set


def _deprecate_method(*, version: str, message: Optional[str] = None):
    """Decorator to issue warnings when using a deprecated method.

    Args:
        version (`str`):
            The version when deprecated arguments will result in error.
        message (`str`, *optional*):
            Warning message that is raised. If not passed, a default warning message
            will be created.
    """

    def _inner_deprecate_method(f):
        @wraps(f)
        def inner_f(*args, **kwargs):
            warning_message = (
                f"'{f.__name__}' (from '{f.__module__}') is deprecated and will be"
                f" removed from version '{version}'."
            )
            if message is not None:
                warning_message += " " + message
            warnings.warn(warning_message, FutureWarning)
            return f(*args, **kwargs)

        return inner_f

    return _inner_deprecate_method