#!/usr/bin/env -S bin/conda/run.sh python

"""
List registered NASA MAAP algorithms.
"""

import sys
from maap.maap import MAAP  # type: ignore

maap = MAAP()
response = maap.listAlgorithms().json()

if response["code"] != 200:
    print("ERROR:", response["message"], file=sys.stderr)
    exit(1)

algorithms = sorted(
    (algorithm["type"], algorithm["version"]) for algorithm in response["algorithms"]
)

print("\n".join(f"{name}:{version}" for name, version in algorithms))
