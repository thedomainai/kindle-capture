#!/usr/bin/env python3
"""E2E diagnostic tests for kindle_capture.

Run with Kindle open and a book displayed:
    python3 test_e2e.py

Tests are sequential and interactive — each test describes what it does
before executing, and reports pass/fail with diagnostic details.
"""

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
from kindle_capture import (
    find_kindle_window, is_kindle_running, activate_kindle,
    capture_window, pixel_hash, capture_and_hash, wait_for_stable_frame,
    send_page_turn, _send_page_back, _safe_remove,
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


def main():
    print("=" * 60)
    print("Kindle Capture — E2E Diagnostic Tests")
    print("=" * 60)
    print()

    # ------------------------------------------------------------------
    # Test 1: Kindle process
    # ------------------------------------------------------------------
    print("[1/8] Kindle process running")
    running = is_kindle_running()
    report("Process check", running,
           "Kindle is running" if running else "Kindle not found — start Kindle first")
    if not running:
        print("\nCannot continue without Kindle. Exiting.")
        sys.exit(1)

    # ------------------------------------------------------------------
    # Test 2: Window detection
    # ------------------------------------------------------------------
    print("\n[2/8] Window detection")
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
    print("\n[3/8] Screenshot capture")
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
    print("\n[4/8] Pixel hash (bitmap decode)")
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
    print("\n[5/8] Pixel hash stability (two captures, 0.5s apart)")
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
    print("\n[6/8] Frame stabilization (wait_for_stable_frame)")
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
    # Test 7: Page turn changes content
    # ------------------------------------------------------------------
    print("\n[7/8] Page turn (right arrow key)")
    before = wait_for_stable_frame(wid, timeout=3.0)
    if not before:
        report("Page turn", False, "could not get stable frame before turn")
    else:
        send_page_turn()
        time.sleep(1.0)
        after = wait_for_stable_frame(wid, timeout=4.0)
        if after and after != before:
            report("Content changed after turn", True)
            # Turn back
            _send_page_back()
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
    # Test 8: Multiple consecutive page turns
    # ------------------------------------------------------------------
    print("\n[8/8] Multiple page turns (3 pages forward, 3 back)")
    hashes = []
    base = wait_for_stable_frame(wid, timeout=3.0)
    if base:
        hashes.append(base)

    forward_ok = True
    for i in range(3):
        send_page_turn()
        time.sleep(1.0)
        h = wait_for_stable_frame(wid, timeout=4.0)
        if h is None:
            report("Forward turn {}".format(i + 1), False, "stabilization failed")
            forward_ok = False
            break
        if h in hashes:
            # We've seen this page before — we went backwards or looped
            idx = hashes.index(h)
            report("Forward turn {}".format(i + 1), False,
                   "page matches earlier page #{} — backward navigation?".format(idx))
            forward_ok = False
            break
        hashes.append(h)
        report("Forward turn {}".format(i + 1), True, "new page (unique hash)")

    if forward_ok:
        # Turn back 3 times
        for i in range(3):
            _send_page_back()
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
