import csv
from pathlib import Path

src = Path('results/runs.csv')
backup = src.with_suffix('.csv.bak')
src.rename(backup)

kept, dropped = [], []
with open(backup, newline='') as f:
    reader = csv.DictReader(f)
    fields = reader.fieldnames
    for row in reader:
        note = (row.get('notes') or '').strip().lower()
        if 'from_friend' in note:
            dropped.append(row); continue
        if 'quick' in note:
            dropped.append(row); continue
        if float(row.get('wall_clock_s') or 0) < 30:
            dropped.append(row); continue
        kept.append(row)

# Keep only the most recent row per (method, dataset, seed)
latest = {}
for r in kept:
    k = (r['run_name'], r['dataset'], r['seed'])
    if k not in latest or r['timestamp'] > latest[k]['timestamp']:
        latest[k] = r

with open(src, 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    w.writerows(latest.values())

print(f'kept {len(latest)} unique rows, dropped {len(dropped)} stale/from_friend rows')
print(f'backup saved to {backup}')
