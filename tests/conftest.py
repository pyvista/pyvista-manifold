"""Shared fixtures for pyvista-manifold tests."""

import pytest
import pyvista as pv

import pyvista_manifold as pvm  # noqa: F401  registers accessor

pv.OFF_SCREEN = True


@pytest.fixture(autouse=True)
def set_default_theme():
    """Reset the testing theme for every test."""
    pv.global_theme.load_theme(pv.plotting.themes._TestingTheme())
    yield
    pv.global_theme.load_theme(pv.plotting.themes._TestingTheme())
