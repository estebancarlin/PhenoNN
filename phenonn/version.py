#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Version information for PhenoNN.
"""

__version__ = "0.1.0"


def get_version():
    """Return the version string."""
    return __version__


def get_versions():
    """Return a dictionary with version information."""
    return {
        "version": __version__,
        "author": "Kazem Ardaneh",
        "email": "kazem.arrdaneh@gmail.com",
        "url": "https://github.com/kardaneh/PhenoNN",
    }
