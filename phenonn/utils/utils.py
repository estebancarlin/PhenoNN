# Copyright 2026 IPSL / CNRS / Sorbonne University
# Authors: Kazem Ardaneh
#
# This work is licensed under the Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc-sa/4.0/


"""
File and lightweight configuration utilities.

This module provides helper classes for simplified file system operations
and convenient dictionary handling with attribute-style access. It is
designed to reduce boilerplate code when working with directories, files,
and configuration-like data structures in Python projects.

The module includes:

- ``EasyDict``: A dictionary subclass enabling attribute-style access
  (e.g., ``cfg.key`` instead of ``cfg["key"]``).
- ``FileUtils``: Static utility methods for creating directories and files.

Features
--------

- Attribute-style access to dictionary keys
- Minimal and dependency-free implementation
- Safe directory creation (no error if directory already exists)
- Simple file creation utility
- Lightweight and suitable for configuration management

Notes
-----

- ``EasyDict`` raises ``AttributeError`` when accessing missing keys,
  making it behave more like standard Python objects.
- ``FileUtils.makedir`` will create nested directories if needed.
- ``FileUtils.makefile`` creates an empty file if it does not exist,
  and does nothing if it already exists.

Dependencies
------------

- os
- typing

Examples
--------

Using EasyDict::

    >>> cfg = EasyDict()
    >>> cfg.learning_rate = 0.001
    >>> cfg.batch_size = 32
    >>> print(cfg.learning_rate)
    0.001
    >>> print(cfg["batch_size"])
    32

Using FileUtils::

    >>> FileUtils.makedir("outputs/logs")
    >>> FileUtils.makefile("outputs/logs", "train.log")

Combined usage::

    >>> paths = EasyDict()
    >>> paths.output_dir = "outputs"
    >>> FileUtils.makedir(paths.output_dir)
"""

import os
from typing import Any


class EasyDict(dict):
    """
    A dictionary subclass that allows for attribute-style access to its items.
    This class extends the built-in dict and overrides the __getattr__, __setattr__, and __delattr__ methods to enable accessing dictionary keys as attributes.
    Original work: Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.
    Original source: https://github.com/NVlabs/edm
    """

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value

    def __delattr__(self, name: str) -> None:
        del self[name]


class FileUtils:
    """
    Utility class for file and directory operations.
    """

    def __init__(self):
        """
        Initialize the FileUtils class. This class does not maintain any state, so the constructor is empty.
        """
        super().__init__()

    @staticmethod
    def makedir(dirs):
        """
        Create a directory if it does not exist.

        Parameters
        ----------
        dirs : str
            The path of the directory to be created.
        """
        if not os.path.exists(dirs):
            os.makedirs(dirs)

    @staticmethod
    def makefile(dirs, filename):
        """
        Create an empty file in the specified directory.
        Parameters
        ----------
        dirs : str
            The path of the directory where the file will be created.
        filename : str
            The name of the file to be created.
        """
        filepath = os.path.join(dirs, filename)
        with open(filepath, "a"):
            pass
