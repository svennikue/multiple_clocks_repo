#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Free disk space in the first-level FEAT directories, without touching the GLMs
that are currently in use.

A first-level .feat directory is ~99% stuff nobody ever reads again: the
voxelwise confound EVs FEAT copies in (confoundEV*/InputconfoundEV*), the
thresholded/rendered zstats, the cluster tables, and a tsplot/ folder with a few
thousand pngs. Everything downstream in this project only ever opens
    <glm>.feat/stats/pe*.nii.gz
(checked: every glm_*.feat path in the repo points at .feat/stats).

TWO LEVELS, as asked for
    Level 1 -- GLMs in active use. Never touched, not even listed for deletion:
        * name matches PROTECTED_GLMS below (01, 01-TR*, instr_*,
          all-paths-fixed_stickrews_split-buttons)
        * OR anything written in the last PROTECT_DAYS days (default 7). This
          also protects FEAT jobs that are running right now.
    Level 2 -- older GLMs. The directory stays, but everything outside KEEP
        (see below) is deleted. The PEs survive, so the RSA can still read them.

WHAT SURVIVES IN A PRUNED GLM (KEEP)
    stats/pe*.nii.gz          the only data anything reads
    stats/dof, smoothness     tiny, tells you the fit actually finished
    design.{fsf,mat,con,...}  tiny, the provenance of the GLM
    absbrainthresh.txt
    custom_timing_files/      tiny text, what went into the design
    --minimal drops all but the PEs.

Nothing outside sub-*/func/glm_*.feat is ever considered: preproc_clean_*.feat,
EVs_*, motion/, anat/ are invisible to this script.

USAGE
    # 1. look first: what would this gain, and where?
    python prune_feat_dirs.py

    # 2. then delete, from the manifest that scan wrote (asks for confirmation)
    python prune_feat_dirs.py --delete /home/fs0/xpsy1114/scratch/data/derivatives/group/feat_cleanup_<stamp>

    # useful variants
    python prune_feat_dirs.py --subjects 01 02 --top 40
    python prune_feat_dirs.py --protect-days 14 --minimal
    python prune_feat_dirs.py --delete <dir> --dry-run

The scan writes settings.json, report.txt, scan.json and to_delete.txt into
derivatives/group/feat_cleanup_<timestamp>/. to_delete.txt is the exact list of
files that --delete will remove: what you read is what gets deleted.

@author: Svenja Kuechenhoff
"""

import argparse
import fnmatch
import json
import os
import stat
import sys
import time
from concurrent.futures import ThreadPoolExecutor

# --- where things live (same detection as check_GLMs_ran.py) --------------
source_dir = "/Users/xpsy1114/Documents/projects/multiple_clocks"
if os.path.isdir(source_dir):
    data_dir_deriv = f"{source_dir}/data/derivatives"
else:
    source_dir = "/home/fs0/xpsy1114/scratch"
    data_dir_deriv = f"{source_dir}/data/derivatives"

DEFAULT_SUBJECTS = [f"{i:02}" for i in range(1, 36) if i not in (21, 29)]

# --- LEVEL 1: GLMs in active use ------------------------------------------
# matched against the glm name, i.e. the NAME in glm_<NAME>_pt0<half>.feat
PROTECTED_GLMS = [
    '01',
    '01-TR*',
    'instr_*',
    'all-paths-fixed_stickrews_split-buttons',
]
PROTECT_DAYS = 7          # anything written this recently is in use / still running

# --- LEVEL 2: what survives in an old GLM ---------------------------------
PRUNE_MARKER = 'PRUNED.json'
KEEP = [
    'stats/pe*.nii.gz',                      # the only thing the RSA reads
    'stats/dof', 'stats/smoothness',
    'design.fsf', 'design.mat', 'design.con',
    'design.frf', 'design.min', 'design.trg',
    'absbrainthresh.txt',
    'custom_timing_files/*',
    PRUNE_MARKER,
]
KEEP_MINIMAL = ['stats/pe*.nii.gz', PRUNE_MARKER]


def human(n):
    """Bytes -> something you can read."""
    for unit in ['B', 'K', 'M', 'G', 'T']:
        if abs(n) < 1024 or unit == 'T':
            return f"{n:.1f}{unit}" if unit != 'B' else f"{int(n)}B"
        n /= 1024.0


def glm_name_of(dirname):
    """'glm_01-TR3_pt02.feat' -> '01-TR3'. Also handles FEAT's '+' twins."""
    name = dirname[len('glm_'):].rsplit('.feat', 1)[0].rstrip('+')
    if '_pt' in name:
        name = name.rsplit('_pt', 1)[0]
    return name


def is_protected_name(glm):
    for pattern in PROTECTED_GLMS:
        if fnmatch.fnmatch(glm, pattern):
            return pattern
    return None


def keep_file(rel, keep_globs):
    return any(fnmatch.fnmatch(rel, pattern) for pattern in keep_globs)


def scan_feat_dir(path, keep_globs):
    """One walk over a .feat directory: how big it is, how much of that we keep,
    when it was last written, and exactly which files would go."""
    total = kept = 0
    newest = os.path.getmtime(path)
    doomed = []
    for root, dirs, files in os.walk(path):
        rel_root = os.path.relpath(root, path)
        for fname in files:
            fpath = os.path.join(root, fname)
            try:
                st = os.lstat(fpath)
            except OSError:
                continue
            if not stat.S_ISREG(st.st_mode):
                continue      # symlinks etc: never deleted, never count as 'recent'
            newest = max(newest, st.st_mtime)
            rel = fname if rel_root == '.' else f"{rel_root}/{fname}"
            total += st.st_size
            if keep_file(rel, keep_globs):
                kept += st.st_size
            else:
                doomed.append(fpath)
    return total, kept, newest, doomed


def scan_subject(sub, keep_globs, protect_days):
    """Every glm_*.feat of one subject, classified."""
    func = f"{data_dir_deriv}/sub-{sub}/func"
    if not os.path.isdir(func):
        return []
    cutoff = time.time() - protect_days * 86400
    out = []
    for dirname in sorted(os.listdir(func)):
        if not (dirname.startswith('glm_') and dirname.endswith('.feat')):
            continue
        path = f"{func}/{dirname}"
        if not os.path.isdir(path):
            continue
        glm = glm_name_of(dirname)
        pattern = is_protected_name(glm)
        if pattern:                           # level 1: do not even walk it
            out.append(dict(sub=sub, dir=dirname, glm=glm, path=path,
                            protected=True, reason=f"name matches '{pattern}'",
                            total=0, keep=0, reclaim=0, n_files=0, doomed=[]))
            continue
        total, kept, newest, doomed = scan_feat_dir(path, keep_globs)
        if os.path.exists(f"{path}/{PRUNE_MARKER}"):
            with open(f"{path}/{PRUNE_MARKER}") as f:
                when = json.load(f).get('pruned', '?')
            out.append(dict(sub=sub, dir=dirname, glm=glm, path=path,
                            protected=True, reason=f"already pruned {when}",
                            total=total, keep=total, reclaim=0, n_files=0, doomed=[]))
            continue
        if newest > cutoff:
            out.append(dict(sub=sub, dir=dirname, glm=glm, path=path,
                            protected=True,
                            reason=f"written {(time.time()-newest)/86400:.1f} days ago",
                            total=total, keep=total, reclaim=0, n_files=0, doomed=[]))
            continue
        out.append(dict(sub=sub, dir=dirname, glm=glm, path=path,
                        protected=False, reason='',
                        total=total, keep=kept, reclaim=total - kept,
                        n_files=len(doomed), doomed=doomed))
    return out


def scan(subjects, keep_globs, protect_days, jobs):
    """All subjects, a few in parallel: the walk is waiting on the file system,
    not on the CPU."""
    rows = []
    with ThreadPoolExecutor(max_workers=jobs) as pool:
        futures = {pool.submit(scan_subject, sub, keep_globs, protect_days): sub
                   for sub in subjects}
        for fut, sub in futures.items():
            found = fut.result()
            rows.extend(found)
            print(f"  sub-{sub}: {len(found)} glm_*.feat, "
                  f"{human(sum(r['reclaim'] for r in found))} reclaimable", flush=True)
    rows.sort(key=lambda r: (r['sub'], r['dir']))
    return rows


def report(rows, protect_days, keep_globs, top):
    """The 'where are the gains' table. Aggregated by GLM, then by subject,
    then the biggest single directories."""
    lines = []
    add = lines.append
    prunable = [r for r in rows if not r['protected']]
    protected = [r for r in rows if r['protected']]

    add(f"FEAT prune scan  --  {time.strftime('%Y-%m-%d %H:%M')}")
    add(f"root            : {data_dir_deriv}")
    add(f"subjects        : {len(set(r['sub'] for r in rows))}")
    add(f"glm_*.feat dirs : {len(rows)}  ({len(protected)} protected, {len(prunable)} prunable)")
    add(f"protected names : {', '.join(PROTECTED_GLMS)}")
    add(f"protected age   : anything written in the last {protect_days} days")
    add(f"kept in pruned  : {', '.join(keep_globs)}")
    add("")

    add("--- LEVEL 1: protected, not touched ---")
    by_glm = {}
    for r in protected:
        why = (is_protected_name(r['glm'])
               or ('already pruned' if r['reason'].startswith('already pruned')
                   else f"written < {protect_days} days ago"))
        by_glm.setdefault((r['glm'], why), []).append(r)
    for (glm, why), rs in sorted(by_glm.items()):
        add(f"  {glm:<52} {len(rs):>4} dirs   ({why})")
    add("")

    add(f"--- LEVEL 2: prunable, biggest gains first ---")
    add(f"  {'GLM':<52} {'dirs':>5} {'total':>9} {'keep':>9} {'reclaim':>9}")
    agg = {}
    for r in prunable:
        a = agg.setdefault(r['glm'], dict(n=0, total=0, keep=0, reclaim=0, files=0))
        a['n'] += 1
        a['total'] += r['total']
        a['keep'] += r['keep']
        a['reclaim'] += r['reclaim']
        a['files'] += r['n_files']
    for glm, a in sorted(agg.items(), key=lambda kv: -kv[1]['reclaim']):
        add(f"  {glm:<52} {a['n']:>5} {human(a['total']):>9} "
            f"{human(a['keep']):>9} {human(a['reclaim']):>9}")
    add(f"  {'TOTAL':<52} {len(prunable):>5} {human(sum(r['total'] for r in prunable)):>9} "
        f"{human(sum(r['keep'] for r in prunable)):>9} "
        f"{human(sum(r['reclaim'] for r in prunable)):>9}")
    add("")

    add("--- reclaim per subject ---")
    per_sub = {}
    for r in prunable:
        per_sub[r['sub']] = per_sub.get(r['sub'], 0) + r['reclaim']
    for sub, gain in sorted(per_sub.items(), key=lambda kv: -kv[1]):
        add(f"  sub-{sub}   {human(gain):>9}")
    add("")

    add(f"--- {top} biggest single directories ---")
    for r in sorted(prunable, key=lambda r: -r['reclaim'])[:top]:
        add(f"  {human(r['reclaim']):>9}  {r['n_files']:>5} files  "
            f"sub-{r['sub']}/{r['dir']}")
    add("")
    add(f"TOTAL RECLAIMABLE: {human(sum(r['reclaim'] for r in prunable))} "
        f"in {sum(r['n_files'] for r in prunable)} files")
    return "\n".join(lines)


def write_scan(rows, args, keep_globs, protect_days):
    """Everything the delete step needs, in derivatives/group/."""
    stamp = time.strftime('%Y%m%d-%H%M%S')
    out = f"{data_dir_deriv}/group/feat_cleanup_{stamp}"
    os.makedirs(out, exist_ok=True)

    settings = dict(created=stamp, script=os.path.abspath(__file__),
                    data_dir_deriv=data_dir_deriv, subjects=args.subjects,
                    protected_glms=PROTECTED_GLMS, protect_days=protect_days,
                    keep_globs=keep_globs, minimal=args.minimal,
                    n_dirs=len(rows),
                    n_prunable=sum(1 for r in rows if not r['protected']),
                    n_files_to_delete=sum(r['n_files'] for r in rows),
                    bytes_reclaimable=sum(r['reclaim'] for r in rows))
    with open(f"{out}/settings.json", 'w') as f:
        json.dump(settings, f, indent=2)
    with open(f"{out}/scan.json", 'w') as f:
        json.dump([{k: v for k, v in r.items() if k != 'doomed'} for r in rows],
                  f, indent=2)
    with open(f"{out}/to_delete.txt", 'w') as f:
        for r in rows:
            for p in r['doomed']:
                f.write(p + "\n")
    return out


# --- deleting -------------------------------------------------------------

def safe_to_delete(path, keep_globs, protect_days, checked_dirs):
    """Re-check every single path at delete time, independently of the manifest.
    A path only goes if it is a real file, inside sub-*/func/glm_*.feat, not in
    KEEP, and its GLM is still not protected."""
    if not (os.path.isabs(path) and path.startswith(data_dir_deriv + '/sub-')):
        return False, 'outside derivatives/sub-*'
    if '..' in path.split('/') or '/func/glm_' not in path or '.feat/' not in path:
        return False, 'not inside a func/glm_*.feat'
    feat = path.split('.feat/')[0] + '.feat'
    rel = path[len(feat) + 1:]
    if keep_file(rel, keep_globs):
        return False, 'on the keep list'
    if feat not in checked_dirs:
        glm = glm_name_of(os.path.basename(feat))
        pattern = is_protected_name(glm)
        if pattern:
            checked_dirs[feat] = f"protected name '{pattern}'"
        else:
            newest = max(os.path.getmtime(feat),
                         os.path.getmtime(f"{feat}/stats")
                         if os.path.isdir(f"{feat}/stats") else 0)
            checked_dirs[feat] = (None if newest < time.time() - protect_days * 86400
                                  else 'written since the scan')
    if checked_dirs[feat]:
        return False, checked_dirs[feat]
    if os.path.islink(path) or not os.path.isfile(path):
        return False, 'not a regular file'
    return True, ''


def delete(cleanup_dir, args):
    global data_dir_deriv
    with open(f"{cleanup_dir}/settings.json") as f:
        settings = json.load(f)
    data_dir_deriv = settings['data_dir_deriv']
    keep_globs = settings['keep_globs']
    protect_days = args.protect_days if args.protect_days is not None else settings['protect_days']

    with open(f"{cleanup_dir}/to_delete.txt") as f:
        paths = [line.strip() for line in f if line.strip()]

    print(f"manifest        : {cleanup_dir}/to_delete.txt")
    print(f"scanned         : {settings['created']}")
    print(f"files listed    : {len(paths)}")
    print(f"space to free   : {human(settings['bytes_reclaimable'])}")
    print(f"kept everywhere : {', '.join(keep_globs)}")
    print(f"protected GLMs  : {', '.join(settings['protected_glms'])} "
          f"+ anything younger than {protect_days} days")
    if args.dry_run:
        print("\n--dry-run: nothing will be deleted.")
    elif not args.yes:
        if input("\nType DELETE to remove these files: ").strip() != 'DELETE':
            print("aborted, nothing deleted.")
            return

    freed = n_del = 0
    skipped = {}
    touched = {}
    checked_dirs = {}
    for path in paths:
        ok, why = safe_to_delete(path, keep_globs, protect_days, checked_dirs)
        if not ok:
            skipped[why] = skipped.get(why, 0) + 1
            continue
        size = os.path.getsize(path)
        if not args.dry_run:
            try:
                os.remove(path)
            except OSError as e:
                skipped[str(e)] = skipped.get(str(e), 0) + 1
                continue
        freed += size
        n_del += 1
        feat = path.split('.feat/')[0] + '.feat'
        touched[feat] = touched.get(feat, 0) + size
        if n_del % 20000 == 0:
            print(f"  ... {n_del} files, {human(freed)}", flush=True)

    if not args.dry_run:
        for feat, size in touched.items():
            for root, dirs, files in os.walk(feat, topdown=False):
                if root != feat and not os.listdir(root):
                    os.rmdir(root)          # tsplot/, logs/ once emptied
            with open(f"{feat}/{PRUNE_MARKER}", 'w') as f:
                json.dump(dict(pruned=time.strftime('%Y-%m-%d %H:%M'),
                               freed_bytes=size, kept=keep_globs,
                               by=os.path.basename(__file__),
                               manifest=cleanup_dir), f, indent=2)
        with open(f"{cleanup_dir}/deleted.log", 'w') as f:
            f.write(f"{time.strftime('%Y-%m-%d %H:%M')}  deleted {n_del} files, "
                    f"freed {human(freed)} in {len(touched)} .feat dirs\n")
            for feat, size in sorted(touched.items(), key=lambda kv: -kv[1]):
                f.write(f"{human(size):>9}  {feat}\n")
        settings['deleted'] = dict(when=time.strftime('%Y-%m-%d %H:%M'),
                                   n_files=n_del, bytes_freed=freed,
                                   n_feat_dirs=len(touched))
        with open(f"{cleanup_dir}/settings.json", 'w') as f:
            json.dump(settings, f, indent=2)

    verb = "would free" if args.dry_run else "freed"
    print(f"\n{verb} {human(freed)} by removing {n_del} files "
          f"from {len(touched)} .feat directories")
    for why, n in skipped.items():
        print(f"  skipped {n} paths: {why}")
    if not args.dry_run:
        print(f"log: {cleanup_dir}/deleted.log")


def main():
    global data_dir_deriv
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--delete', metavar='CLEANUP_DIR',
                   help='delete the files listed in a previous scan (asks first)')
    p.add_argument('--subjects', nargs='+', default=DEFAULT_SUBJECTS)
    p.add_argument('--root', default=data_dir_deriv,
                   help=f'derivatives directory (default {data_dir_deriv})')
    p.add_argument('--protect-days', type=int, default=None,
                   help=f'leave GLMs written this recently alone (default {PROTECT_DAYS})')
    p.add_argument('--minimal', action='store_true',
                   help='keep only stats/pe*.nii.gz, not the small design files')
    p.add_argument('--top', type=int, default=25,
                   help='how many single directories to list (default 25)')
    p.add_argument('--jobs', type=int, default=4,
                   help='subjects scanned in parallel (default 4)')
    p.add_argument('--yes', action='store_true', help='skip the confirmation prompt')
    p.add_argument('--dry-run', action='store_true',
                   help='with --delete: say what would go, delete nothing')
    args = p.parse_args()

    data_dir_deriv = args.root.rstrip('/')

    if args.delete:
        delete(args.delete.rstrip('/'), args)
        return

    protect_days = args.protect_days if args.protect_days is not None else PROTECT_DAYS
    keep_globs = KEEP_MINIMAL if args.minimal else KEEP
    print(f"scanning {len(args.subjects)} subjects under {data_dir_deriv} ...", flush=True)
    rows = scan(args.subjects, keep_globs, protect_days, args.jobs)
    text = report(rows, protect_days, keep_globs, args.top)
    print("\n" + text)

    out = write_scan(rows, args, keep_globs, protect_days)
    with open(f"{out}/report.txt", 'w') as f:
        f.write(text + "\n")
    print(f"\nwritten to {out}")
    print(f"review  : less {out}/to_delete.txt")
    print(f"delete  : python {os.path.basename(__file__)} --delete {out}")


if __name__ == '__main__':
    main()
