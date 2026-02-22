# Available at setup time due to pyproject.toml

import subprocess

from setuptools import setup

__version__ = "4.1.0"

CCLIB_PATH = "coolchic/CCLIB"

subprocess.call(f"rm -rf {CCLIB_PATH}/*", shell=True)
subprocess.call("rm -rf coolchic/coolchic.egg-info/*", shell=True)

# C++ extensions disabled for this setup (no C++ sources).
ext_modules = []

setup(
    name="coolchic",
    version=__version__,
    author="Orange",
    author_email="theo.ladune@orange.com",
    url="https://github.com/Orange-OpenSource/Cool-Chic",
    description="Cool-Chic: lightweight neural video codec.",
    long_description="",
    ext_modules=ext_modules,
    extras_require={},
    zip_safe=False,
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.7.1",
        "torchvision",
        "matplotlib",
        "einops",
        "fvcore",
        "cmake",
        "ConfigArgParse",
        "psutil",
        "pytest",
        "pytest-order",
        "scipy",
        "opencv-python",
        "torchac",
        "pandas",
        "tyro",
    ],
)

subprocess.call(f"mkdir -p {CCLIB_PATH}", shell=True)
subprocess.call(f"mkdir -p {CCLIB_PATH}", shell=True)
