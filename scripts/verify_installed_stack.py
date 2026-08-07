#!/usr/bin/env python3
"""Verify that an installed environment matches the intended ABI stack.

Used by CI after ``uv sync`` to check what actually landed on disk, rather than
what the lockfile claims.

Two modes:

``--expect-cuda``
    A Linux GPU install. The three compiled extensions must be present and
    their local versions must name the CUDA channel and PyTorch minor that
    ``pyproject.toml`` selects.

``--expect-no-cuda``
    A CPU install (notably Apple Silicon). None of the CUDA-only packages may
    be installed, and torch must not be a CUDA build.

Neither mode imports torch's CUDA runtime or requires a GPU to be present.
"""

from __future__ import annotations

import argparse
import importlib.metadata as md
import re
import sys
import tomllib
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

CRITICAL_EXTENSIONS = ("flash-attn", "mamba-ssm", "causal-conv1d")

#: Packages that must never appear in a CPU-only environment.
CUDA_ONLY_PACKAGES = (
    *CRITICAL_EXTENSIONS,
    "xformers",
    "triton",
    "nvidia-cutlass-dsl",
    "tilelang",
    "quack-kernels",
)


def _expected_stack() -> tuple[str, str]:
    """Return the ``(cuda, torch)`` version strings pyproject selects."""
    pyproject = tomllib.loads(
        (PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    )

    url = next(
        index["url"]
        for index in pyproject["tool"]["uv"]["index"]
        if index["name"] == "astral-cu130"
    )
    channel = re.search(r"/cu(\d+)(\d)/?$", url)
    assert channel is not None, f"unrecognised Astral index URL: {url}"

    torch_req = next(
        r
        for r in pyproject["project"]["optional-dependencies"]["gpu"]
        if r.startswith("torch~=")
    )
    torch = re.search(r"~=\s*(\d+)\.(\d+)", torch_req)
    assert torch is not None

    return (
        f"{channel.group(1)}.{channel.group(2)}",
        f"{torch.group(1)}.{torch.group(2)}",
    )


def _installed() -> dict[str, str]:
    versions = {}
    for dist in md.distributions():
        name = dist.metadata["Name"]
        if name:
            versions[re.sub(r"[-_.]+", "-", name).lower()] = dist.version
    return versions


def check_cuda_install(required: list[str]) -> list[str]:
    """Check the CUDA stack.

    Every *installed* extension must carry the expected local version; the
    packages named in ``required`` must additionally be present. Which
    extensions an extra pulls in differs -- ``gpu`` has no flash-attn, since
    only the gigapath/conch1_5/musk variants in ``gpu_all`` need it.
    """
    cuda, torch_minor = _expected_stack()
    expected_local = f"+cu.{cuda}.torch.{torch_minor}"
    installed = _installed()
    errors: list[str] = []

    for name in CRITICAL_EXTENSIONS:
        version = installed.get(name)
        if version is None:
            if name in required:
                errors.append(f"{name} is required but not installed")
            else:
                print(f"  --  {name} not installed (not required by this extra)")
        elif expected_local not in version:
            errors.append(
                f"{name}=={version} does not carry the expected local version "
                f"{expected_local!r}"
            )
        else:
            print(f"  ok  {name}=={version}")

    import torch

    if torch.version.cuda is None:
        errors.append("torch is not a CUDA build")
    elif not torch.version.cuda.startswith(cuda):
        errors.append(f"torch reports CUDA {torch.version.cuda}, expected {cuda}")
    else:
        print(f"  ok  torch=={torch.__version__} (CUDA {torch.version.cuda})")

    if not torch.__version__.startswith(torch_minor):
        errors.append(
            f"torch {torch.__version__} does not match the pinned {torch_minor} series"
        )

    return errors


def check_cpu_install() -> list[str]:
    installed = _installed()
    errors = [
        f"{name}=={installed[name]} must not be installed in a CPU environment"
        for name in CUDA_ONLY_PACKAGES
        if name in installed
    ]

    leaked = sorted(
        name
        for name in installed
        if name.startswith(("nvidia-", "cuda-")) or name == "cudnn"
    )
    errors.extend(
        f"{name} must not be installed in a CPU environment" for name in leaked
    )

    import torch

    if torch.version.cuda is not None:
        errors.append(f"torch reports CUDA {torch.version.cuda}, expected a CPU build")
    else:
        print(f"  ok  torch=={torch.__version__} (CPU build)")

    if not errors:
        print(f"  ok  none of {', '.join(CUDA_ONLY_PACKAGES)} are installed")

    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--expect-cuda", action="store_true")
    group.add_argument("--expect-no-cuda", action="store_true")
    parser.add_argument(
        "--require",
        nargs="*",
        default=list(CRITICAL_EXTENSIONS),
        metavar="PACKAGE",
        help="extensions that must be installed (only with --expect-cuda)",
    )
    args = parser.parse_args()

    errors = (
        check_cuda_install(args.require) if args.expect_cuda else check_cpu_install()
    )

    if errors:
        print(f"\n{len(errors)} problem(s) with the installed environment:\n")
        for error in errors:
            print(f"  * {error}")
        return 1

    print("\ninstalled environment matches the intended stack.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
