#!/usr/bin/env conda run -n sar2-d2 python

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
longest_name = max(len(name) for name, _ in algorithms)

print("\n".join(f"{name}: {version}" for name, version in algorithms))
