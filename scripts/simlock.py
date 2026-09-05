#!/usr/bin/env python3
"""Machine-wide slot lock so simulations across worktrees queue instead of colliding.

Every v2ecoli worktree on this machine shares one physical RAM budget. A
multi-generation sim peaks around 5.5 GB, so a 16 GB laptop fits two, and a
third pushes the machine into swap where everything runs slower than if the
third had simply waited. Nothing enforced that, so it was managed by hand --
badly, because occupancy changes while you are not looking.

This holds the registry OUTSIDE any worktree (~/.v2ecoli/simlock/) so every
checkout sees the same state.

Use it as a wrapper -- no runner changes needed:

    python3 scripts/simlock.py run -- \
        .venv/bin/python3 scripts/run_condition_multigen_parquet.py --cache-dir ...

It blocks until a slot frees, runs the command, and releases the slot on exit,
including on Ctrl-C or SIGTERM. Other subcommands:

    python3 scripts/simlock.py status         # who holds what, and for how long
    python3 scripts/simlock.py adopt 1234     # put an already-running sim under the lock
    python3 scripts/simlock.py wait           # block until a slot is free, then exit 0
    python3 scripts/simlock.py reap           # drop entries whose process is gone

Capacity defaults to 2 and is overridable with V2ECOLI_SIM_SLOTS. Set
V2ECOLI_SIM_MIN_FREE_GB (default 3) to also refuse to start when the machine is
already short of memory regardless of slot count -- slots assume every sim is a
full-size one, and that guard catches the case where something else large is
running.
"""
from __future__ import annotations

import argparse
import errno
import fcntl
import json
import os
import signal
import socket
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path

ROOT = Path(os.environ.get("V2ECOLI_SIMLOCK_DIR", Path.home() / ".v2ecoli" / "simlock"))
REGISTRY = ROOT / "slots.json"
LOCKFILE = ROOT / ".registry.lock"
POLL_SECONDS = 15


def capacity() -> int:
    try:
        return max(1, int(os.environ.get("V2ECOLI_SIM_SLOTS", "2")))
    except ValueError:
        return 2


def min_free_gb() -> float:
    try:
        return float(os.environ.get("V2ECOLI_SIM_MIN_FREE_GB", "3"))
    except ValueError:
        return 3.0


def free_gb() -> float | None:
    """Free + inactive memory in GB, or None if it cannot be determined."""
    try:
        out = subprocess.run(["vm_stat"], capture_output=True, text=True, timeout=5).stdout
        page = 4096
        first = out.splitlines()[0]
        if "page size of" in first:
            page = int(first.split("page size of")[1].split()[0])
        stats = {}
        for line in out.splitlines()[1:]:
            if ":" in line:
                k, v = line.split(":", 1)
                stats[k.strip()] = int(v.strip().rstrip("."))
        pages = stats.get("Pages free", 0) + stats.get("Pages inactive", 0)
        return pages * page / 1024**3
    except Exception:
        return None


def alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except OSError as exc:
        return exc.errno == errno.EPERM
    return True


@contextmanager
def registry_lock():
    ROOT.mkdir(parents=True, exist_ok=True)
    fd = os.open(LOCKFILE, os.O_CREAT | os.O_RDWR, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


def _read() -> list[dict]:
    if not REGISTRY.exists():
        return []
    try:
        return json.loads(REGISTRY.read_text())
    except (json.JSONDecodeError, OSError):
        return []


def _write(entries: list[dict]) -> None:
    tmp = REGISTRY.with_suffix(".tmp")
    tmp.write_text(json.dumps(entries, indent=1))
    tmp.replace(REGISTRY)


def _reap(entries: list[dict]) -> list[dict]:
    return [e for e in entries if alive(e.get("pid", -1))]


def try_claim(label: str, command: str) -> bool:
    """Claim a slot if one is free. Caller must hold no lock."""
    with registry_lock():
        entries = _reap(_read())
        if len(entries) >= capacity():
            _write(entries)
            return False
        fg = free_gb()
        if fg is not None and fg < min_free_gb() and entries:
            # Slots free but the machine is not. Only block when something else
            # is already running -- otherwise a single sim could never start.
            _write(entries)
            return False
        entries.append({
            "pid": os.getpid(), "host": socket.gethostname(),
            "worktree": str(Path.cwd()), "label": label, "command": command,
            "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "started_epoch": time.time(),
        })
        _write(entries)
        return True


def release() -> None:
    with registry_lock():
        _write([e for e in _read() if e.get("pid") != os.getpid()])


def cmd_status(_args) -> int:
    with registry_lock():
        entries = _reap(_read())
        _write(entries)
    fg = free_gb()
    print(f"slots: {len(entries)}/{capacity()} in use"
          + (f"   free memory: {fg:.1f} GB (floor {min_free_gb():.1f})" if fg is not None else ""))
    if not entries:
        print("  (idle)")
    for e in entries:
        mins = (time.time() - e.get("started_epoch", time.time())) / 60
        print(f"  pid {e['pid']:<7} {mins:6.1f} min  {e.get('label','?')}")
        print(f"      {e.get('worktree','?')}")
    return 0


def cmd_adopt(args) -> int:
    """Register an already-running process as holding a slot.

    Needed whenever sims were started before the lock existed, or from a
    checkout that predates it: simlock would otherwise see free slots that the
    machine does not actually have, which is worse than no lock at all.
    """
    claimed, skipped = [], []
    with registry_lock():
        entries = _reap(_read())
        held = {e.get("pid") for e in entries}
        for pid in args.pids:
            if not alive(pid):
                skipped.append((pid, "not running"))
                continue
            if pid in held:
                skipped.append((pid, "already held"))
                continue
            try:
                cmdline = subprocess.run(["ps", "-p", str(pid), "-o", "command="],
                                         capture_output=True, text=True, timeout=5).stdout.strip()
            except Exception:
                cmdline = ""
            entries.append({
                "pid": pid, "host": socket.gethostname(), "worktree": "(adopted)",
                "label": args.label or (cmdline.split()[1] if len(cmdline.split()) > 1
                                        else f"pid-{pid}").split("/")[-1],
                "command": cmdline, "adopted": True,
                "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "started_epoch": time.time(),
            })
            held.add(pid)
            claimed.append(pid)
        _write(entries)
    for pid in claimed:
        print(f"adopted pid {pid}")
    for pid, why in skipped:
        print(f"skipped pid {pid}: {why}")
    if len(entries) > capacity():
        print(f"note: {len(entries)} holders now exceed capacity {capacity()}; "
              f"new runs will queue until it drops", file=sys.stderr)
    return 0


def cmd_reap(_args) -> int:
    with registry_lock():
        before = _read()
        after = _reap(before)
        _write(after)
    print(f"reaped {len(before) - len(after)} dead entr"
          f"{'y' if len(before) - len(after) == 1 else 'ies'}; {len(after)} live")
    return 0


def _wait_for_slot(label: str, command: str, timeout: float | None) -> bool:
    deadline = None if timeout is None else time.time() + timeout
    announced = False
    while True:
        if try_claim(label, command):
            return True
        if not announced:
            with registry_lock():
                held = _reap(_read())
            who = ", ".join(f"pid {e['pid']} ({e.get('label','?')})" for e in held) or "unknown"
            print(f"[simlock] all {capacity()} slots busy -- waiting. Held by: {who}",
                  file=sys.stderr, flush=True)
            announced = True
        if deadline is not None and time.time() > deadline:
            return False
        time.sleep(POLL_SECONDS)


def cmd_wait(args) -> int:
    ok = _wait_for_slot(args.label or "wait", "", args.timeout)
    if ok:
        release()
    return 0 if ok else 1


def cmd_run(args) -> int:
    if not args.command:
        print("simlock run: nothing to run (use -- before the command)", file=sys.stderr)
        return 2
    label = args.label or Path(args.command[0]).name
    cmdline = " ".join(args.command)
    if args.no_wait:
        if not try_claim(label, cmdline):
            print("[simlock] no free slot and --no-wait given; not starting", file=sys.stderr)
            return 75  # EX_TEMPFAIL
    else:
        if not _wait_for_slot(label, cmdline, args.timeout):
            print(f"[simlock] timed out after {args.timeout}s waiting for a slot", file=sys.stderr)
            return 75
    print(f"[simlock] slot acquired ({label})", file=sys.stderr, flush=True)

    proc: subprocess.Popen | None = None

    def forward(signum, _frame):
        if proc is not None:
            proc.send_signal(signum)

    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, forward)
    try:
        proc = subprocess.Popen(args.command)
        return proc.wait()
    finally:
        release()
        print("[simlock] slot released", file=sys.stderr, flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(prog="simlock", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_run = sub.add_parser("run", help="acquire a slot, run a command, release")
    p_run.add_argument("--label", default=None, help="short name shown in status")
    p_run.add_argument("--timeout", type=float, default=None, help="seconds to wait for a slot")
    p_run.add_argument("--no-wait", action="store_true", help="fail immediately if no slot is free")
    p_run.add_argument("command", nargs=argparse.REMAINDER)
    p_run.set_defaults(func=cmd_run)

    p_wait = sub.add_parser("wait", help="block until a slot is free, then exit")
    p_wait.add_argument("--label", default=None)
    p_wait.add_argument("--timeout", type=float, default=None)
    p_wait.set_defaults(func=cmd_wait)

    p_adopt = sub.add_parser("adopt", help="register already-running pids as slot holders")
    p_adopt.add_argument("pids", nargs="+", type=int)
    p_adopt.add_argument("--label", default=None)
    p_adopt.set_defaults(func=cmd_adopt)

    sub.add_parser("status", help="show slot occupancy").set_defaults(func=cmd_status)
    sub.add_parser("reap", help="drop entries whose process is gone").set_defaults(func=cmd_reap)

    args = ap.parse_args()
    if getattr(args, "command", None) and args.command and args.command[0] == "--":
        args.command = args.command[1:]
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
