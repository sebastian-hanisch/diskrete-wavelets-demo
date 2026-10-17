"""Gemeinsame Fixtures: die Mini-Instanz, an der jede Einheit von Hand geprüft wird (DEMO-PLAYBOOK §1b)."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@pytest.fixture
def mini_x():
    """Haar (h = (1, 1)/sqrt 2, g = (1, -1)/sqrt 2), x = (1, 2, 3, 4): Stufe 1: d = ((1 - 2), (3 - 4))/sqrt 2 = (-1/sqrt 2, -1/sqrt 2), a = ((1 + 2), (3 + 4))/sqrt 2 = (3/sqrt 2, 7/sqrt 2);
    Stufe 2: d2 = (a0 - a1)/sqrt 2 = (3 - 7)/2 = -2, a2 = (3 + 7)/2 = 5; Energie 1 + 1 + 4 + 25 = 30 = 1 + 4 + 9 + 16."""
    return np.array([1.0, 2.0, 3.0, 4.0])
