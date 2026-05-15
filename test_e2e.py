#!/usr/bin/env python3
"""E2E diagnostic tests for kindle_capture.

Run with Kindle open and a book displayed:
    python3 test_e2e.py
    python3 test_e2e.py --direction ltr    # for horizontal books
    python3 test_e2e.py --direction rtl    # for manga/vertical books

Tests are sequential and interactive — each test describes what it does
before executing, and reports pass/fail with diagnostic details.
"""

import argparse
import hashlib
import os
import subprocess
import sys
import tempfile
import time

try:
    import Quartz.CoreGraphics as CG
except ImportError:
    print("FATAL: Quartz not available.", file=sys.stderr)
    sys.exit(1)

# Import functions from kindle_capture
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import kindle_capture as kc
from kindle_capture import (
    find_kindle_window, is_kindle_running, activate_kindle,
    capture_window, pixel_hash, capture_and_hash, wait_for_stable_frame,
    send_page_turn, _send_page_back, _safe_remove,
    probe_navigation_keys,
)

PASS = 0
FAIL = 0
WARN = 0


def report(name, ok, detail="", warn=False):
    global PASS, FAIL, WARN
    if warn:
        WARN += 1
        print("  WARN  {} — {}".format(name, detail))
    elif ok:
        PASS += 1
        print("  OK    {}{}".format(name, " — " + detail if detail else ""))
    else:
        FAIL += 1
        print("  FAIL  {} — {}".format(name, detail))


def parse_args():
    parser = argparse.ArgumentParser(description="E2E diagnostic tests for kindle_capture")
    parser.add_argument("--direction", choices=["auto", "rtl", "ltr"], default="auto",
                        help="Reading direction (default: auto)")
    parser.add_argument("--verbose", action="store_true",
                        help="Print debug information")
    return parser.parse_args()


def main():
    args = parse_args()
    direction = args.direction

    print("=" * 60)
    print("Kindle Capture — E2E Diagnostic Tests")
    print("  direction: {}".format(direction))
    print("=" * 60)
    print()

    # ------------------------------------------------------------------
    # Test 1: Kindle process
    # ------------------------------------------------------------------
    print("[1/9] Kindle process running")
    running = is_kindle_running()
    report("Process check", running,
           "Kindle is running" if running else "Kindle not found — start Kindle first")
    if not running:
        print("\nCannot continue without Kindle. Exiting.")
        sys.exit(1)

    # ------------------------------------------------------------------
    # Test 2: Window detection
    # ------------------------------------------------------------------
    print("\n[2/9] Window detection")
    window = find_kindle_window()
    if window:
        report("Window found", True,
               "id={} size={}x{}".format(window["id"], window["width"], window["height"]))
    else:
        report("Window found", False, "No visible Kindle window")
        print("\nCannot continue without a window. Exiting.")
        sys.exit(1)
    wid = window["id"]

    # ------------------------------------------------------------------
    # Test 3: Screenshot capture
    # ------------------------------------------------------------------
    print("\n[3/9] Screenshot capture")
    fd, tmp = tempfile.mkstemp(suffix=".png")
    os.close(fd)
    ok = capture_window(wid, tmp)
    if ok:
        size = os.path.getsize(tmp)
        report("screencapture -l", True, "{} bytes".format(size))
    else:
        report("screencapture -l", False, "Command failed or no file produced")
    _safe_remove(tmp)

    # ------------------------------------------------------------------
    # Test 4: Pixel hash decode
    # ------------------------------------------------------------------
    print("\n[4/9] Pixel hash (bitmap decode)")
    fd, tmp = tempfile.mkstemp(suffix=".png")
    os.close(fd)
    capture_window(wid, tmp)
    h1 = pixel_hash(tmp)
    if h1:
        report("Decode attempt 1", True, "hash={}".format(h1[:16]))
    else:
        report("Decode attempt 1", False, "pixel_hash returned None")

    # Decode same file again — must be identical
    h2 = pixel_hash(tmp)
    report("Deterministic decode", h1 == h2,
           "same" if h1 == h2 else "DIFFERENT (h1={} h2={})".format(
               h1[:16] if h1 else "None", h2[:16] if h2 else "None"))
    _safe_remove(tmp)

    # ------------------------------------------------------------------
    # Test 5: Pixel hash stability across captures (same window, no action)
    # ------------------------------------------------------------------
    print("\n[5/9] Pixel hash stability (two captures, 0.5s apart)")
    tmp1, hash1 = capture_and_hash(wid)
    time.sleep(0.5)
    tmp2, hash2 = capture_and_hash(wid)
    if tmp1:
        _safe_remove(tmp1)
    if tmp2:
        _safe_remove(tmp2)

    if hash1 and hash2:
        if hash1 == hash2:
            report("Cross-capture stability", True, "identical pixel hashes")
        else:
            report("Cross-capture stability", False,
                   "hashes differ (cursor blink? animation?)",
                   warn=True)
    else:
        report("Cross-capture stability", False,
               "capture failed (h1={} h2={})".format(hash1, hash2))

    # ------------------------------------------------------------------
    # Test 6: Frame stabilization
    # ------------------------------------------------------------------
    print("\n[6/9] Frame stabilization (wait_for_stable_frame)")
    t0 = time.time()
    stable = wait_for_stable_frame(wid, timeout=4.0, interval=0.3)
    elapsed = time.time() - t0
    if stable:
        report("Stabilization", True,
               "hash={} in {:.1f}s".format(stable[:16], elapsed))
    else:
        report("Stabilization", False,
               "timeout after {:.1f}s — window content keeps changing".format(elapsed))

    # ------------------------------------------------------------------
    # Test 7: Navigation key probing
    # ------------------------------------------------------------------
    print("\n[7/9] Navigation key probe (direction={})".format(direction))
    activate_kindle()
    time.sleep(0.5)
    nav_result = probe_navigation_keys(wid, direction=direction, verbose=args.verbose)
    nav_ok = True
    if nav_result:
        fwd, bwd, name = nav_result
        report("Navigation probe", True,
               "forward=key {}, backward=key {}, method={}".format(fwd, bwd, name))
    else:
        if direction == "auto":
            report("Navigation probe", False,
                   "both arrows change page — direction ambiguous. "
                   "Use --direction rtl or --direction ltr")
            nav_ok = False
        else:
            report("Navigation probe", True,
                   "using {} arrow keys (no probe needed)".format(direction.upper()))

    # ------------------------------------------------------------------
    # Test 8: Page turn changes content
    # ------------------------------------------------------------------
    if not nav_ok:
        print("\n[8/9] Page turn — SKIPPED (direction ambiguous)")
        print("[9/9] Multiple page turns — SKIPPED (direction ambiguous)")
        print("\n  Tip: Navigate Kindle to the FIRST page for auto-detection,")
        print("       or specify --direction rtl / --direction ltr")
    else:
        nav_desc = kc._nav_key_name if kc._nav_key_name else (
            "left arrow (RTL)" if direction == "rtl" else "right arrow (LTR)")
        print("\n[8/9] Page turn ({})".format(nav_desc))
        before = wait_for_stable_frame(wid, timeout=3.0)
        if not before:
            report("Page turn", False, "could not get stable frame before turn")
        else:
            send_page_turn(direction=direction)
            time.sleep(1.0)
            after = wait_for_stable_frame(wid, timeout=4.0)
            if after and after != before:
                report("Content changed after turn", True)
                # Turn back
                _send_page_back(direction=direction)
                time.sleep(1.0)
                restored = wait_for_stable_frame(wid, timeout=3.0)
                if restored == before:
                    report("Restored after turn-back", True)
                else:
                    report("Restored after turn-back", False,
                           "hash differs from original", warn=True)
            else:
                report("Content changed after turn", False,
                       "page did not change — is a book open?")

        # ------------------------------------------------------------------
        # Test 9: Multiple consecutive page turns
        # ------------------------------------------------------------------
        print("\n[9/9] Multiple page turns (3 pages forward, 3 back)")
        hashes = []
        base = wait_for_stable_frame(wid, timeout=3.0)
        if base:
            hashes.append(base)

        forward_ok = True
        for i in range(3):
            send_page_turn(direction=direction)
            time.sleep(1.0)
            h = wait_for_stable_frame(wid, timeout=4.0)
            if h is None:
                report("Forward turn {}".format(i + 1), False, "stabilization failed")
                forward_ok = False
                break
            if h in hashes:
                idx = hashes.index(h)
                report("Forward turn {}".format(i + 1), False,
                       "page matches earlier page #{} — backward navigation?".format(idx))
                forward_ok = False
                break
            hashes.append(h)
            report("Forward turn {}".format(i + 1), True, "new page (unique hash)")

        if forward_ok:
            for i in range(3):
                _send_page_back(direction=direction)
                time.sleep(1.0)
            restored = wait_for_stable_frame(wid, timeout=3.0)
            if restored == base:
                report("Returned to original page", True)
            else:
                report("Returned to original page", False,
                       "hash differs", warn=True)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Results: {} passed, {} failed, {} warnings".format(PASS, FAIL, WARN))
    print("=" * 60)

    if FAIL > 0:
        print("\nSome tests failed. Fix the issues above before running kindle_capture.py.")
        sys.exit(1)
    elif WARN > 0:
        print("\nWarnings detected. Capture may work but check the details above.")
    else:
        print("\nAll checks passed. Ready to capture.")


if __name__ == "__main__":
    main()
