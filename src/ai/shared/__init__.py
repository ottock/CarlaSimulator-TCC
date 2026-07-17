"""Preprocessing shared between simulator (PC) and car (Jetson).

IMPORTANT: keep this subpackage Python 3.6-compatible (no walrus operator, no
dataclasses, no structural pattern matching). It is the single source of truth
for how raw sensor data becomes model input, so sim and real must run byte-for-byte
identical code.
"""
