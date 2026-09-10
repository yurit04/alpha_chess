from __future__ import annotations

"""Locate, build and load the native self-play core (``_fastchess``).

The tree search is the throughput ceiling of the whole project: profiling the
pure-Python engine puts ~40% of self-play in python-chess move generation, ~14%
in move indexing and ~10% in board encoding, which together leave an RTX 3090
around a quarter busy.  ``alpha_chess/native/`` reimplements exactly that hot
path -- bitboard move generation, the encoding, and the PUCT tree itself -- as a
CPython extension, leaving Python with only the GPU hand-off.

The extension is compiled on demand the first time it is needed and cached
next to the package, so a fresh clone needs no build step.  Any C compiler will
do; when the host has none, ``pip install ziglang`` provides one that needs no
root.  Everything degrades to the pure-Python engine if no compiler is found,
so this module never raises on import.
"""

import os
import subprocess
import sys
import sysconfig
import threading
from typing import List, Optional

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC_DIR = os.path.join(_HERE, "native")
_SOURCES = ["fastchess.c"]
_HEADERS = ["bitboard.h", "position.h", "encode.h", "mcts.h"]

_lock = threading.Lock()
_module = None
_attempted = False
_error: Optional[str] = None


def _ext_path() -> str:
    suffix = sysconfig.get_config_var("EXT_SUFFIX") or ".so"
    return os.path.join(_HERE, "_fastchess" + suffix)


def _sources_newer_than(target: str) -> bool:
    """True when the extension is missing or older than any of its sources."""
    if not os.path.exists(target):
        return True
    built = os.path.getmtime(target)
    for name in _SOURCES + _HEADERS:
        path = os.path.join(_SRC_DIR, name)
        if os.path.exists(path) and os.path.getmtime(path) > built:
            return True
    return False


def _compiler_commands() -> List[List[str]]:
    """Compiler front-ends to try, best first.

    ``CC`` wins if set; otherwise the usual system compilers, then ziglang,
    which is a pip-installable clang and so works without root.
    """
    candidates: List[List[str]] = []
    env_cc = os.environ.get("CC")
    if env_cc:
        candidates.append(env_cc.split())
    for name in ("cc", "gcc", "clang"):
        candidates.append([name])
    candidates.append([sys.executable, "-m", "ziglang", "cc"])
    return candidates


def build(verbose: bool = False, force: bool = False) -> Optional[str]:
    """Compile the extension; return its path, or ``None`` if no compiler works."""
    target = _ext_path()
    if not force and not _sources_newer_than(target):
        return target

    include = sysconfig.get_paths()["include"]
    sources = [os.path.join(_SRC_DIR, s) for s in _SOURCES]
    tmp_target = target + ".tmp{0}".format(os.getpid())
    base_flags = [
        "-O3", "-DNDEBUG", "-fPIC", "-shared", "-fno-strict-aliasing",
        "-fvisibility=hidden", "-I" + include, "-I" + _SRC_DIR,
    ]
    # -march=native is worth having (popcount/bit-scan/pext-class instructions
    # matter a lot here) but is dropped rather than failing the build on hosts
    # or front-ends that reject it.
    attempts = [["-march=native"], []]

    errors = []
    for cc in _compiler_commands():
        for arch in attempts:
            cmd = cc + base_flags + arch + sources + ["-o", tmp_target, "-lm"]
            try:
                proc = subprocess.run(
                    cmd, capture_output=True, text=True, timeout=600
                )
            except (OSError, subprocess.SubprocessError) as exc:
                errors.append("{0}: {1}".format(cc[0], exc))
                break
            if proc.returncode == 0:
                os.replace(tmp_target, target)
                if verbose:
                    print("Built native self-play core: {0}".format(target))
                return target
            errors.append(
                "{0}{1} failed:\n{2}".format(
                    " ".join(cc), " " + " ".join(arch) if arch else "",
                    (proc.stderr or proc.stdout)[-2000:],
                )
            )
    try:
        os.remove(tmp_target)
    except OSError:
        pass
    global _error
    _error = "\n".join(errors[-2:]) or "no C compiler found"
    if verbose:
        print("Could not build the native core:\n{0}".format(_error))
    return None


def load(verbose: bool = False):
    """Import the native core, building it once if needed. ``None`` on failure."""
    global _module, _attempted
    with _lock:
        if _module is not None:
            return _module
        if _attempted:
            return None
        _attempted = True
        try:
            if _sources_newer_than(_ext_path()):
                build(verbose=verbose)
            import importlib

            mod = importlib.import_module("alpha_chess._fastchess")
            _check_constants(mod)
            _module = mod
            return mod
        except Exception as exc:  # pragma: no cover - depends on the toolchain
            global _error
            _error = "{0}: {1}".format(type(exc).__name__, exc)
            if verbose:
                print("Native core unavailable ({0}); using the Python engine."
                      .format(_error))
            return None


def _check_constants(mod) -> None:
    """Fail loudly if the C and Python sides disagree on a shared constant.

    These describe the layout of arrays that cross the boundary raw, so a
    mismatch would not raise -- it would silently produce garbage training data.
    """
    from alpha_chess.batched_selfplay import MAX_POLICY_TARGETS
    from alpha_chess.encoding import NUM_PLANES, POLICY_SIZE

    expected = {
        "NUM_PLANES": NUM_PLANES,
        "POLICY_SIZE": POLICY_SIZE,
        "MAX_POLICY_TARGETS": MAX_POLICY_TARGETS,
    }
    for name, want in expected.items():
        got = getattr(mod, name, None)
        if got != want:
            raise RuntimeError(
                "native core was built with {n}={g}, but this build of "
                "alpha_chess uses {w}. Rebuild it with "
                "`python -m alpha_chess.native --force`.".format(
                    n=name, g=got, w=want
                )
            )


def available() -> bool:
    """Whether the native core can be used in this process."""
    return load() is not None


def last_error() -> Optional[str]:
    """Why the native core is unavailable, if it is."""
    return _error


def describe() -> str:
    """One-line status string for the training log."""
    mod = load()
    if mod is not None:
        return "native core: {0}".format(os.path.basename(mod.__file__))
    return "native core unavailable ({0})".format(_error or "unknown")


if __name__ == "__main__":  # pragma: no cover - manual build entry point
    path = build(verbose=True, force="--force" in sys.argv)
    if path is None:
        sys.exit(1)
    mod = load(verbose=True)
    print("OK" if mod is not None else "built but not importable")
