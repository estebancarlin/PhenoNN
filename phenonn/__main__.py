#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Allow running the PhenoNN package as a module with: python -m phenonn
"""

import sys
from .cli import main

if __name__ == "__main__":
    sys.exit(main())
