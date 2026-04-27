"""Shared fixtures for pyvista-manifold tests."""

from collections.abc import Generator
import importlib

import pytest
import pyvista as pv

importlib.import_module('pyvista_manifold')

pv.OFF_SCREEN = True


@pytest.fixture(autouse=True)
def set_default_theme() -> Generator[None, None, None]:
    """Reset the testing theme for every test."""
    pv.global_theme.load_theme(pv.plotting.themes._TestingTheme())
    yield
    pv.global_theme.load_theme(pv.plotting.themes._TestingTheme())
