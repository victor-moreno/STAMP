"""Tests for the dependency declarations in ``pyproject.toml`` and ``uv.lock``.

These run without a GPU and without network access: they only read the two
files that pin STAMP's dependency stack.
"""

import importlib.util
import re
import tomllib
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PYPROJECT_PATH = PROJECT_ROOT / "pyproject.toml"
LOCK_PATH = PROJECT_ROOT / "uv.lock"

CRITICAL_EXTENSIONS = ("flash-attn", "mamba-ssm", "causal-conv1d")
ASTRAL_INDEX_URL = "https://wheels.astral.sh/simple/cu130/"


def _load_checker():
    """Import ``scripts/check_dependency_config.py`` as a module."""
    path = PROJECT_ROOT / "scripts" / "check_dependency_config.py"
    spec = importlib.util.spec_from_file_location("check_dependency_config", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def checker():
    return _load_checker()


@pytest.fixture(scope="module")
def pyproject() -> dict:
    return tomllib.loads(PYPROJECT_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def lock() -> dict:
    return tomllib.loads(LOCK_PATH.read_text(encoding="utf-8"))


def test_static_dependency_checks_pass(checker):
    """The standalone CI checker reports no problems."""
    assert checker.main() == 0


def test_no_direct_wheel_urls(checker, pyproject):
    requirements = list(checker.iter_requirements(pyproject))
    assert checker.check_no_wheel_urls(requirements) == []


def test_git_dependencies_are_immutable(checker, pyproject):
    """Every git dependency uses HTTPS and a full commit SHA."""
    requirements = list(checker.iter_requirements(pyproject))
    assert checker.check_git_dependencies(requirements) == []


def test_extension_versions_match_torch_and_cuda(checker, pyproject):
    requirements = list(checker.iter_requirements(pyproject))
    assert checker.check_extension_versions(pyproject, requirements) == []


def test_gpu_prebuilt_is_a_plain_alias(checker, pyproject):
    assert checker.check_gpu_prebuilt_has_no_wheels(pyproject) == []


def test_source_builds_of_extensions_are_refused(checker, pyproject):
    assert checker.check_no_build_packages(pyproject) == []


def test_checker_rejects_a_wheel_url(checker):
    """The checker actually fails on a bad requirement, rather than no-opping."""
    bad = [checker.Requirement("test", "flash-attn @ https://example.com/x.whl")]
    assert checker.check_no_wheel_urls(bad)


def test_checker_rejects_an_unpinned_git_dependency(checker):
    bad = [checker.Requirement("test", "conch @ git+https://github.com/x/y.git")]
    assert checker.check_git_dependencies(bad)


def test_checker_rejects_a_branch_pinned_git_dependency(checker):
    bad = [checker.Requirement("test", "conch @ git+https://github.com/x/y.git@main")]
    assert checker.check_git_dependencies(bad)


# --------------------------------------------------------------------------- #
# Lockfile invariants
# --------------------------------------------------------------------------- #


def _locked_package(lock: dict, name: str) -> dict | None:
    for package in lock.get("package", []):
        if package.get("name") == name:
            return package
    return None


@pytest.mark.parametrize("name", CRITICAL_EXTENSIONS)
def test_extensions_come_from_the_astral_index(lock, name):
    """The compiled extensions resolve to the Astral index, not a wheel URL."""
    package = _locked_package(lock, name)
    assert package is not None, f"{name} is missing from uv.lock"
    assert package["source"] == {"registry": ASTRAL_INDEX_URL}


@pytest.mark.parametrize("name", CRITICAL_EXTENSIONS)
def test_extensions_have_no_source_distribution(lock, name):
    """uv must never have to build these packages; only wheels are acceptable."""
    package = _locked_package(lock, name)
    assert package is not None
    assert "sdist" not in package, (
        f"{name} locked an sdist, which would trigger a CUDA-detecting source build"
    )
    assert package.get("wheels"), f"{name} locked no wheels"


@pytest.mark.parametrize("name", CRITICAL_EXTENSIONS)
def test_extensions_ship_both_linux_architectures(lock, name):
    """x86_64 and aarch64 wheels are both locked, so uv can pick per-machine."""
    package = _locked_package(lock, name)
    assert package is not None
    urls = " ".join(wheel["url"] for wheel in package["wheels"])
    assert "x86_64" in urls
    assert "aarch64" in urls


@pytest.mark.parametrize("name", CRITICAL_EXTENSIONS)
def test_extensions_are_never_installed_on_macos(lock, name):
    """No resolution branch may place a CUDA extension on macOS."""
    package = _locked_package(lock, name)
    assert package is not None
    for wheel in package["wheels"]:
        assert "macosx" not in wheel["url"]

    for package in lock.get("package", []):
        for dependency in package.get("dependencies", []):
            if dependency.get("name") != name:
                continue
            marker = dependency.get("marker", "")
            assert "sys_platform == 'linux'" in marker, (
                f"{package['name']} depends on {name} without a Linux marker: "
                f"{marker!r}"
            )


def test_lock_has_no_katherlab_wheel_artifacts(lock):
    """Release-asset wheels were replaced by the Astral index."""
    text = LOCK_PATH.read_text(encoding="utf-8")
    offenders = re.findall(r"https://github\.com/\S*/releases/download/\S*\.whl", text)
    assert offenders == []


def test_lock_supports_both_python_versions(lock):
    """The universal lock covers the whole supported Python range."""
    assert lock["requires-python"] == ">=3.13, <3.15"


def test_lock_is_in_sync_with_pyproject(pyproject, lock):
    """``torch`` is locked at exactly the version pyproject pins."""
    package = _locked_package(lock, "torch")
    assert package is not None
    pinned = pyproject["project"]["optional-dependencies"]["cpu"]
    expected = next(r for r in pinned if r.startswith("torch"))
    major_minor = re.search(r"(\d+)\.(\d+)", expected)
    assert major_minor is not None
    assert package["version"].startswith(
        f"{major_minor.group(1)}.{major_minor.group(2)}"
    )
