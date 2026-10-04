#!/usr/bin/env python3
"""Verify unmodified locally pinned upstream artifacts and their release lock."""
from face_intel.spec import verify_pin

if __name__ == "__main__":
    pin = verify_pin()
    print(f"StarIntel {pin['schemaVersion']} verified at {pin['commit']}")
