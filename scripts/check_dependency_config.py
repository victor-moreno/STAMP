#!/usr/bin/env python3
"""Static consistency checks for STAMP's dependency declarations.

Runs on ``pyproject.toml`` alone, using nothing but the standard library, so CI
can execute it before (or entirely without) installing the project.

The checks encode decisions that are easy to regress by hand-editing
``pyproject.toml``:

* compiled CUDA extensions must come from an index, never a pinned wheel URL,
  so that uv can pick the right Python ABI and architecture;
* git dependencies must be immutable (HTTPS + full commit SHA);
* the ``flash-attn`` / ``mamba-ssm`` / ``causal-conv1d`` local versions must
  agree with the PyTorch minor and CUDA channel the project actually selects.

Exit status is 0 when clean and 1 when any check fails.
"""

from __future__ import annotations

import re
import sys
import tomllib
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PYPROJECT = PROJECT_ROOT / "pyproject.toml"

#: Packages whose ABI is tied to a specific PyTorch build and CUDA toolkit.
CRITICAL_EXTENSIONS = frozenset({"flash-attn", "mamba-ssm", "causal-conv1d"})

#: Name of the uv index that serves the pre-built CUDA extension wheels.
ASTRAL_INDEX_NAME = "astral-cu130"

_FULL_SHA = re.compile(r"^[0-9a-f]{40}$")

# ``flash-attn==2.8.3.post1+cu.13.0.torch.2.11`` and friends.
_LOCAL_VERSION = re.compile(
    r"\+cu\.(?P<cuda_major>\d+)\.(?P<cuda_minor>\d+)"
    r"\.torch\.(?P<torch_major>\d+)\.(?P<torch_minor>\d+)$"
)

# ``name[extras] <specifier> ; marker`` -- we only need the leading name.
_REQUIREMENT_NAME = re.compile(r"^\s*(?P<name>[A-Za-z0-9._-]+)")


def _normalize(name: str) -> str:
    """Normalize a distribution name per PEP 503."""
    return re.sub(r"[-_.]+", "-", name).lower()


@dataclass(frozen=True)
class Requirement:
    """A single requirement string plus where it was declared."""

    origin: str
    text: str

    @property
    def name(self) -> str:
        match = _REQUIREMENT_NAME.match(self.text)
        return _normalize(match.group("name")) if match else ""


def iter_requirements(pyproject: dict) -> Iterator[Requirement]:
    """Yield every requirement string declared anywhere in ``pyproject``."""
    project = pyproject.get("project", {})

    for text in project.get("dependencies", []):
        yield Requirement("project.dependencies", text)

    for extra, texts in project.get("optional-dependencies", {}).items():
        for text in texts:
            yield Requirement(f"project.optional-dependencies.{extra}", text)

    for group, texts in pyproject.get("dependency-groups", {}).items():
        for text in texts:
            if isinstance(text, str):
                yield Requirement(f"dependency-groups.{group}", text)

    uv = pyproject.get("tool", {}).get("uv", {})
    for key in ("override-dependencies", "constraint-dependencies"):
        for text in uv.get(key, []):
            yield Requirement(f"tool.uv.{key}", text)


def check_no_wheel_urls(requirements: list[Requirement]) -> list[str]:
    """Direct ``.whl`` URLs hard-code Python ABI, architecture and libc."""
    return [
        f"{req.origin}: direct wheel URL is not allowed -- use an index "
        f"requirement instead: {req.text!r}"
        for req in requirements
        if ".whl" in req.text
    ]


def check_git_dependencies(requirements: list[Requirement]) -> list[str]:
    """Git dependencies must be HTTPS and pinned to a full commit SHA."""
    errors: list[str] = []
    for req in requirements:
        if "git+" not in req.text:
            continue

        if "git+http://" in req.text:
            errors.append(
                f"{req.origin}: git dependency must use https, not http: {req.text!r}"
            )

        # Everything after the URL's ``@`` separator, ignoring the ``git+ssh``
        # style ``user@host`` prefix and any trailing environment marker.
        url = req.text.split("git+", 1)[1].split(";", 1)[0].strip()
        _scheme, _, remainder = url.partition("://")
        revision = remainder.rpartition("@")[2] if "@" in remainder else ""

        if not _FULL_SHA.match(revision):
            errors.append(
                f"{req.origin}: git dependency must be pinned to a full 40-character "
                f"commit SHA (got {revision or '<no revision>'!r}): {req.text!r}"
            )
    return errors


def check_url_requirement_markers(requirements: list[Requirement]) -> list[str]:
    """``name @ url`` requirements need whitespace before the marker semicolon.

    PEP 508 requires it, and hatchling enforces it even though uv accepts the
    tighter spelling -- so omitting the space still locks fine but breaks
    ``uv build`` and any install that builds STAMP's own metadata.
    """
    return [
        f"{req.origin}: URL requirement needs whitespace before the ';' marker "
        f"separator (PEP 508): {req.text!r}"
        for req in requirements
        if " @ " in req.text and re.search(r"\S;", req.text)
    ]


def check_gpu_prebuilt_has_no_wheels(pyproject: dict) -> list[str]:
    """``gpu_prebuilt`` is a deprecated alias and must not regain wheel URLs."""
    extras = pyproject.get("project", {}).get("optional-dependencies", {})
    return [
        f"project.optional-dependencies.gpu_prebuilt: must remain a plain alias "
        f"without wheel URLs, found: {text!r}"
        for text in extras.get("gpu_prebuilt", [])
        if ".whl" in text or "https://" in text
    ]


def _configured_cuda_version(pyproject: dict) -> tuple[str, str] | None:
    """Return the ``(major, minor)`` CUDA version of the Astral index URL.

    Channel names spell the minor version as a single trailing digit, so
    ``cu130`` means CUDA 13.0 and ``cu128`` means CUDA 12.8.
    """
    for index in pyproject.get("tool", {}).get("uv", {}).get("index", []):
        if index.get("name") != ASTRAL_INDEX_NAME:
            continue
        match = re.search(r"/cu(\d+)(\d)/?$", index.get("url", ""))
        if match:
            return match.group(1), match.group(2)
    return None


#: Extras that pin the exact torch build the whole environment is bound to.
_TORCH_PINNING_EXTRAS = ("cpu", "gpu", "gpu_all")


def _configured_torch_version(pyproject: dict) -> tuple[str, str] | None:
    """Return the ``(major, minor)`` PyTorch version the blanket extras pin.

    Only the blanket targets count: other extras carry loose floors such as
    ``torch>=2.0.0`` that say nothing about the selected ABI.
    """
    extras = pyproject.get("project", {}).get("optional-dependencies", {})
    found: set[tuple[str, str]] = set()

    for extra in _TORCH_PINNING_EXTRAS:
        for text in extras.get(extra, []):
            if _normalize(Requirement("", text).name) != "torch":
                continue
            match = re.search(r"~=\s*(\d+)\.(\d+)", text)
            if match:
                found.add((match.group(1), match.group(2)))

    if len(found) == 1:
        return found.pop()
    return None


def check_extension_versions(
    pyproject: dict, requirements: list[Requirement]
) -> list[str]:
    """Extension local versions must match the selected torch/CUDA stack."""
    errors: list[str] = []

    cuda = _configured_cuda_version(pyproject)
    if cuda is None:
        return [
            (
                f"tool.uv.index: no index named {ASTRAL_INDEX_NAME!r} with a "
                "recognisable /cuXY/ URL was found"
            )
        ]

    torch = _configured_torch_version(pyproject)
    if torch is None:
        return [
            (
                f"the {', '.join(_TORCH_PINNING_EXTRAS)} extras must each pin torch "
                "with a single agreed `~=MAJOR.MINOR.PATCH` requirement"
            )
        ]

    sources = pyproject.get("tool", {}).get("uv", {}).get("sources", {})
    seen: set[str] = set()

    for req in requirements:
        if req.name not in CRITICAL_EXTENSIONS:
            continue
        seen.add(req.name)

        match = _LOCAL_VERSION.search(req.text.split(";", 1)[0].strip())
        if match is None:
            errors.append(
                f"{req.origin}: {req.name} must be pinned to a full Astral local "
                f"version such as '+cu.{cuda[0]}.{cuda[1]}.torch.{torch[0]}.{torch[1]}': "
                f"{req.text!r}"
            )
            continue

        found_cuda = (match.group("cuda_major"), match.group("cuda_minor"))
        found_torch = (match.group("torch_major"), match.group("torch_minor"))

        if found_cuda != cuda:
            errors.append(
                f"{req.origin}: {req.name} targets CUDA "
                f"{'.'.join(found_cuda)} but the {ASTRAL_INDEX_NAME} index serves CUDA "
                f"{'.'.join(cuda)}: {req.text!r}"
            )
        if found_torch != torch:
            errors.append(
                f"{req.origin}: {req.name} targets torch "
                f"{'.'.join(found_torch)} but the project pins torch "
                f"{'.'.join(torch)}: {req.text!r}"
            )

    for name in sorted(seen):
        if sources.get(name, {}).get("index") != ASTRAL_INDEX_NAME:
            errors.append(
                f"tool.uv.sources: {name} must be pinned to the "
                f"{ASTRAL_INDEX_NAME!r} index"
            )

    return errors


def check_no_build_packages(pyproject: dict) -> list[str]:
    """uv must refuse to build the CUDA extensions from source."""
    configured = {
        _normalize(name)
        for name in pyproject.get("tool", {}).get("uv", {}).get("no-build-package", [])
    }
    missing = CRITICAL_EXTENSIONS - configured
    return [
        f"tool.uv.no-build-package: must list {name!r} so an unsupported platform "
        f"fails loudly instead of running its CUDA-detecting build backend"
        for name in sorted(missing)
    ]


def main() -> int:
    pyproject = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))
    requirements = list(iter_requirements(pyproject))

    errors = [
        *check_no_wheel_urls(requirements),
        *check_git_dependencies(requirements),
        *check_url_requirement_markers(requirements),
        *check_gpu_prebuilt_has_no_wheels(pyproject),
        *check_extension_versions(pyproject, requirements),
        *check_no_build_packages(pyproject),
    ]

    if errors:
        print(f"{len(errors)} dependency configuration problem(s) found:\n")
        for error in errors:
            print(f"  * {error}")
        return 1

    print("pyproject.toml dependency configuration is consistent.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
