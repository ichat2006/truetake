"""Create a new true-origin catalog; never rewrites existing dataset files."""
import csv
from pathlib import Path
import re
import hashlib

ROOT=Path(__file__).resolve().parents[1]

def file_hash(path):
    digest=hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):digest.update(chunk)
    return digest.hexdigest()

def migrate():
    target=ROOT/'data/manifest_v2.csv'
    if target.exists():raise FileExistsError('Refusing to overwrite existing manifest_v2.csv')
    rows=[]
    for name in ['data/manifest.csv','data/clips/variants/manifest.csv','data/clips/variants/overnight_manifest.csv']:
        with (ROOT/name).open(newline='') as handle:rows+=list(csv.DictReader(handle))
    indexed={r['clip_id']:r for r in rows}
    def origin(row):
        if row.get('parent_clip_id'):return origin(indexed[row['parent_clip_id']])
        match=re.search(r'original_source=(\w+)',row.get('notes',''))
        if match:return {'epic_kitchens':'epic'}.get(match[1],match[1])
        return row['source']
    for row in rows:
        row['collection']=row['source'];row['source']=origin(row)
        parent=indexed.get(row.get('parent_clip_id',''))
        modified=False
        if parent:
            a,b=ROOT/row['video_path'],ROOT/parent['video_path']
            modified=a.is_file() and b.is_file() and a.resolve()!=b.resolve() and file_hash(a)!=file_hash(b)
        row['media_modified']='true' if modified else 'false'
    fields=list(dict.fromkeys(key for row in rows for key in row))
    with target.open('x',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=fields);writer.writeheader();writer.writerows(rows)
    print(f'Created {target}: {len(rows)} rows; existing files untouched')

if __name__=='__main__':migrate()
