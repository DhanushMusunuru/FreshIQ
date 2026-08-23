#!/usr/bin/env python
"""
Client side of the FreshIQ <-> Fusion 360 bridge.

The add-in (FreshIQBridge) watches cad/out/inbox and replies into cad/out/outbox.
This script drops a command file and waits for the reply.

    python cad/bridge.py status
    python cad/bridge.py send handshake
    python cad/bridge.py send build --ctx '{"show_dialog": false}'
    python cad/bridge.py send animate --ctx '{"frames": 72}' --timeout 900
    python cad/bridge.py log

Exit code is 0 only when the reply says ok.
"""

import argparse
import json
import os
import sys
import time
import uuid

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")
INBOX = os.path.join(OUT, "inbox")
OUTBOX = os.path.join(OUT, "outbox")
BRIDGE_JSON = os.path.join(OUT, "bridge.json")


def bridge_status():
    if not os.path.exists(BRIDGE_JSON):
        return None
    with open(BRIDGE_JSON, encoding="utf-8") as fh:
        return json.load(fh)


def send(command, ctx, timeout, quiet=False):
    status = bridge_status()
    if status is None:
        print("BRIDGE DOWN - no cad/out/bridge.json.\n"
              "In Fusion: Utilities > ADD-INS > Scripts and Add-Ins >\n"
              "           Add-Ins tab > FreshIQBridge > Run", file=sys.stderr)
        return None

    os.makedirs(INBOX, exist_ok=True)
    os.makedirs(OUTBOX, exist_ok=True)

    job = "{}-{}".format(command, uuid.uuid4().hex[:8])
    reply_path = os.path.join(OUTBOX, job + ".json")

    tmp = os.path.join(INBOX, job + ".part")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump({"command": command, "ctx": ctx}, fh, indent=2)
    os.replace(tmp, os.path.join(INBOX, job + ".json"))

    if not quiet:
        print("-> {} (waiting up to {}s)".format(job, timeout), file=sys.stderr)

    deadline = time.time() + timeout
    while time.time() < deadline:
        if os.path.exists(reply_path):
            time.sleep(0.05)
            with open(reply_path, encoding="utf-8") as fh:
                return json.load(fh)
        time.sleep(0.25)

    print("TIMEOUT after {}s waiting for {}".format(timeout, job), file=sys.stderr)
    return None


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("status")
    sub.add_parser("log")

    s = sub.add_parser("send")
    s.add_argument("command")
    s.add_argument("--ctx", default="{}")
    s.add_argument("--timeout", type=float, default=300.0)
    s.add_argument("--brief", action="store_true",
                   help="print only ok/error, not the whole payload")

    a = ap.parse_args()

    if a.cmd == "status":
        st = bridge_status()
        if st is None:
            print("BRIDGE DOWN")
            return 1
        print(json.dumps(st, indent=2))
        return 0

    if a.cmd == "log":
        p = os.path.join(OUT, "bridge.log")
        if os.path.exists(p):
            with open(p, encoding="utf-8") as fh:
                sys.stdout.write(fh.read())
        else:
            print("(no bridge.log yet)")
        return 0

    reply = send(a.command, json.loads(a.ctx), a.timeout)
    if reply is None:
        return 2

    if a.brief:
        print(json.dumps({k: reply.get(k) for k in
                          ("ok", "command", "seconds", "error")}, indent=2))
    else:
        print(json.dumps(reply, indent=2))
    return 0 if reply.get("ok") else 1


if __name__ == "__main__":
    sys.exit(main())
