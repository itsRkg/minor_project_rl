"""
fix_gym_install.py — patches gym==0.21.0's malformed setup.py and installs it.

Root cause: gym-0.21.0/setup.py contains 'opencv-python>=3.' (trailing dot,
no minor version).  Modern packaging/wheel parses this strictly and rejects it.
This script downloads the tarball, fixes the one bad line, and installs locally.

Run from the rnd_env environment:
    python rnd_baseline/fix_gym_install.py
"""

import subprocess
import sys
import os
import tarfile
import urllib.request
import tempfile
import shutil

GYM_URL = (
    "https://files.pythonhosted.org/packages/source/g/gym/gym-0.21.0.tar.gz"
)
GYM_TARBALL = "gym-0.21.0.tar.gz"
GYM_DIR     = "gym-0.21.0"


def run(cmd, **kwargs):
    print(f"\n>>> {' '.join(cmd)}")
    result = subprocess.run(cmd, **kwargs)
    if result.returncode != 0:
        print(f"ERROR: command failed with exit code {result.returncode}")
        sys.exit(1)
    return result


def main():
    tmp = tempfile.mkdtemp(prefix="gym_patch_")
    print(f"Working directory: {tmp}")

    # ── 1. Download ───────────────────────────────────────────────────────────
    tarpath = os.path.join(tmp, GYM_TARBALL)
    print(f"\nDownloading {GYM_URL} ...")
    urllib.request.urlretrieve(GYM_URL, tarpath)
    print(f"Downloaded → {tarpath}")

    # ── 2. Extract ────────────────────────────────────────────────────────────
    print("\nExtracting ...")
    with tarfile.open(tarpath, "r:gz") as t:
        t.extractall(tmp)
    gym_src = os.path.join(tmp, GYM_DIR)
    assert os.path.isdir(gym_src), f"Expected {gym_src} after extraction"

    # ── 3. Patch setup.py ─────────────────────────────────────────────────────
    setup_path = os.path.join(gym_src, "setup.py")
    with open(setup_path, "r", encoding="utf-8") as f:
        original = f.read()

    # The bad line: "opencv-python>=3."  → fix to "opencv-python>=3.0"
    bad     = "opencv-python>=3."
    fixed   = "opencv-python>=3.0"
    if bad not in original:
        print(f"\nWARNING: '{bad}' not found in setup.py — may already be patched or different version.")
    else:
        patched = original.replace(bad, fixed)
        with open(setup_path, "w", encoding="utf-8") as f:
            f.write(patched)
        print(f"\nPatched setup.py: '{bad}' → '{fixed}'")

    # ── 4. Install gym core from patched source ───────────────────────────────
    # Install without extras first (the [atari] extras pull ale-py which we
    # install separately with a known-good version below).
    run([
        sys.executable, "-m", "pip", "install",
        gym_src,
        "--no-build-isolation",
    ])

    # ── 5. Install Atari support packages ────────────────────────────────────
    # ale-py 0.7.5 is the last version that ships gym-0.21-compatible ROMs
    # and works on Python 3.10 Windows.
    run([
        sys.executable, "-m", "pip", "install",
        "ale-py==0.7.5",
    ])

    # AutoROM installs the actual Atari ROM files
    run([
        sys.executable, "-m", "pip", "install",
        "AutoROM[accept-rom-license]",
    ])

    # ── 6. Cleanup ────────────────────────────────────────────────────────────
    shutil.rmtree(tmp, ignore_errors=True)
    print(f"\nCleaned up {tmp}")

    # ── 7. Verify ─────────────────────────────────────────────────────────────
    print("\n=== Verification ===")
    run([sys.executable, "-c",
         "import gym; print('gym version:', gym.__version__)"])
    run([sys.executable, "-c",
         "import ale_py; print('ale_py version:', ale_py.__version__)"])

    print("\n✓ gym==0.21.0 installed successfully.")
    print("\nNext step — install Atari ROMs (run once, needs Internet):")
    print("    AutoROM --accept-license")


if __name__ == "__main__":
    main()
