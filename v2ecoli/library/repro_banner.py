"""Reproducibility banner for v2ecoli reports.

Returns a small HTML snippet with: date (Eastern time), git commit,
host, user, Python version, and platform. Intended to be injected at
the top of every generated report for traceability.
"""

import datetime
import getpass
import os
import platform
import subprocess
import sys

try:
    from zoneinfo import ZoneInfo
    _ET = ZoneInfo('America/New_York')
except Exception:
    _ET = None


def _git(args, cwd=None):
    try:
        out = subprocess.check_output(
            ['git'] + list(args), cwd=cwd or os.getcwd(),
            stderr=subprocess.DEVNULL, timeout=2)
        return out.decode().strip()
    except Exception:
        return ''


def collect_provenance():
    """Return a dict of reproducibility fields."""
    now = datetime.datetime.now(tz=_ET) if _ET else datetime.datetime.now()
    fmt = '%Y-%m-%d %H:%M:%S %Z' if _ET else '%Y-%m-%d %H:%M:%S (local)'
    commit = _git(['rev-parse', '--short', 'HEAD']) or 'unknown'
    commit_full = _git(['rev-parse', 'HEAD']) or 'unknown'
    branch = _git(['rev-parse', '--abbrev-ref', 'HEAD']) or 'unknown'
    dirty = bool(_git(['status', '--porcelain']))
    remote = _git(['config', '--get', 'remote.origin.url'])
    # Describe the machine: darwin/Linux, chip kind, cores, RAM hint.
    machine = f'{platform.system()} {platform.release()} {platform.machine()}'
    try:
        import multiprocessing
        cores = multiprocessing.cpu_count()
    except Exception:
        cores = '?'
    return {
        'when': now.strftime(fmt),
        'commit': commit,
        'commit_full': commit_full,
        'branch': branch,
        'dirty': dirty,
        'remote': remote,
        'user': getpass.getuser(),
        'host': platform.node(),
        'machine': machine,
        'cores': cores,
        'python': f'Python {sys.version.split()[0]}',
        'platform': platform.platform(),
    }


def banner_html():
    """Return an HTML string suitable for embedding at the top of a report."""
    p = collect_provenance()
    dirty_tag = (' <span style="color:#dc2626;font-weight:600">'
                 '[uncommitted changes]</span>') if p['dirty'] else ''
    return f'''
<div style="font-family:-apple-system,BlinkMacSystemFont,Segoe UI,Roboto,sans-serif;
            background:#f8fafc;border:1px solid #e2e8f0;border-radius:6px;
            padding:10px 14px;margin:0 0 16px 0;font-size:12px;color:#475569;
            line-height:1.6">
  <strong style="color:#0f172a">Run provenance</strong> &nbsp;·&nbsp;
  {p['when']} &nbsp;·&nbsp;
  <code>{p['branch']}@{p['commit']}</code>{dirty_tag} &nbsp;·&nbsp;
  <span title="Host">{p['user']}@{p['host']}</span> &nbsp;·&nbsp;
  {p['machine']} ({p['cores']} cores) &nbsp;·&nbsp;
  {p['python']}
</div>
'''
