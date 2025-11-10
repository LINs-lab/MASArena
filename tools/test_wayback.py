#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import datetime as dt
import json
import os
import sys
import time
from typing import Optional, Tuple

import urllib.parse as urlparse
import urllib.request as urlrequest

# Simple HTTP helper using stdlib to avoid extra deps

def http_get(url: str, timeout: float, use_proxy: bool) -> Tuple[int, bytes]:
    req = urlrequest.Request(url, headers={
        "User-Agent": "WaybackTester/1.0 (+https://example.com)"
    })
    opener = urlrequest.build_opener()
    if not use_proxy:
        opener = urlrequest.build_opener(urlrequest.ProxyHandler({}))
    with opener.open(req, timeout=timeout) as resp:
        code = resp.getcode() or 0
        data = resp.read()
        return code, data


def query_wayback_available(target_url: str, yyyymmdd: str, timeout: float, use_proxy: bool) -> Optional[str]:
    api = "https://archive.org/wayback/available"
    q = urlparse.urlencode({"url": target_url, "timestamp": yyyymmdd})
    url = f"{api}?{q}"
    code, data = http_get(url, timeout=timeout, use_proxy=use_proxy)
    if code != 200:
        return None
    try:
        payload = json.loads(data.decode("utf-8", errors="ignore"))
    except Exception:
        return None
    archived_snap = payload.get("archived_snapshots", {}).get("closest")
    if archived_snap and archived_snap.get("available") and archived_snap.get("url"):
        return archived_snap["url"]
    return None


def fetch_with_retries(url: str, timeout: float, retries: int, backoff: float, use_proxy: bool) -> Tuple[bool, int, int]:
    # Returns (ok, status, size)
    attempt = 0
    while attempt <= retries:
        try:
            code, data = http_get(url, timeout=timeout, use_proxy=use_proxy)
            ok = (200 <= code < 300)
            if ok and data:
                return True, code, len(data)
            # Retry on non-2xx or empty body
        except Exception as e:
            last_err = str(e)
        attempt += 1
        time.sleep(backoff * (2 ** (attempt - 1)))
    # Final attempt failure
    try:
        code, data = http_get(url, timeout=timeout, use_proxy=use_proxy)
        return (200 <= code < 300), code, len(data or b"")
    except Exception:
        return False, 0, 0


def date_str(d: dt.date) -> str:
    return d.strftime("%Y%m%d")


def main():
    parser = argparse.ArgumentParser(description="Test Wayback snapshot availability with retries and proxy control.")
    parser.add_argument("url", help="Target URL to snapshot, e.g., https://en.wikipedia.org/wiki/Mercedes_Sosa")
    parser.add_argument("date", help="Preferred date in YYYYMMDD, e.g., 20221231")
    parser.add_argument("--timeout", type=float, default=15.0, help="HTTP timeout seconds (default: 15)")
    parser.add_argument("--retries", type=int, default=3, help="Retries for fetching snapshot (default: 3)")
    parser.add_argument("--backoff", type=float, default=0.6, help="Exponential backoff base seconds (default: 0.6)")
    parser.add_argument("--no-proxy", action="store_true", help="Bypass proxies for archive.org and target fetch")
    parser.add_argument("--neighbor-window", type=int, default=3, help="Search +/- N days if exact date not available (default: 3)")
    args = parser.parse_args()

    use_proxy = not args.no_proxy

    base_date = dt.datetime.strptime(args.date, "%Y%m%d").date()

    # Probe exact date first, then neighbor days: 0, -1, +1, -2, +2, ...
    offsets = [0]
    for i in range(1, max(0, args.neighbor_window) + 1):
        offsets.extend([-i, i])

    picked_snapshot = None
    picked_when = None

    for off in offsets:
        ymd = date_str(base_date + dt.timedelta(days=off))
        snapshot = query_wayback_available(args.url, ymd, timeout=args.timeout, use_proxy=use_proxy)
        print(f"Probe date {ymd}: {'FOUND' if snapshot else 'NONE'}")
        if snapshot:
            picked_snapshot = snapshot
            picked_when = ymd
            break

    if not picked_snapshot:
        print("No snapshot found within window.")
        sys.exit(2)

    print(f"Using snapshot ({picked_when}): {picked_snapshot}")

    ok, status, size = fetch_with_retries(picked_snapshot, timeout=args.timeout, retries=args.retries, backoff=args.backoff, use_proxy=use_proxy)
    print(f"Fetch snapshot -> ok={ok} status={status} size={size}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()



