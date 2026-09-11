#!/usr/bin/env python3
"""Control preflight adapter: the historical audit and window audit must both pass."""

from audit_window_run import main

if __name__ == "__main__":
    raise SystemExit(main(include_legacy=True))
