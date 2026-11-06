"""Separate, YAML-driven media permissions; never changes a quality verdict."""
from copy import deepcopy
from pathlib import Path
import yaml

ROOT=Path(__file__).resolve().parents[2]
POLICY_PATH=ROOT/'contracts/licenses.yaml'

def policy():
    return yaml.safe_load(POLICY_PATH.read_text())

def permissions(license_id=None):
    rules=policy()
    return deepcopy(rules['licenses'].get(license_id,rules['licenses'][rules['default_license']]))

def apply_license(card,contract=None):
    result=dict(card)
    entry=card.get('manifest_entry') or {}
    rules=policy()
    license_id=card.get('license') or entry.get('license') or rules['default_license']
    if license_id not in rules['licenses']:license_id=rules['default_license']
    defaults=yaml.safe_load((ROOT/'contracts/defaults.yaml').read_text())
    intended=(contract or {}).get('intended_use') or card.get('intended_use') or defaults['intended_use']
    if intended not in rules['intended_uses']:raise ValueError('Unknown intended use')
    rights=permissions(license_id)
    modified=card.get('media_modified',entry.get('media_modified',False))
    modified=str(modified).lower() in {'true','1','yes'}
    privacy_conflict=bool(card.get('privacy_blur_intervals')) and not rights['modify']
    public=rights['share_public'] and rights['show_original'] and not (rights.get('share_original_only') and modified) and not privacy_conflict
    use=rules['intended_uses'][intended]
    eligible=bool(rights[use['permission']])
    label=('Usable for ' if eligible else 'Not eligible for ')+use['label']
    if not rights['train_commercial']:label+=' · Not eligible for commercial training ('+license_id+')'
    reasons=[]
    if not rights['share_public']:reasons.append('Licence disallows public media')
    if rights.get('share_original_only') and modified:reasons.append('Original-only sharing: this media is modified')
    if privacy_conflict:reasons.append('Required privacy blur conflicts with no-modification permission')
    result.update(license=license_id,intended_use=intended,media_modified=modified,
        license_permissions=rights,credit_line=rights['credit_line'],
        eligibility={'eligible':eligible,'intended_use':intended,'line':label},
        public_media_allowed=bool(public),media_display_allowed=bool(rights['show_original'] and not privacy_conflict and (not modified or rights['modify'])),
        public_exclusion='; '.join(reasons))
    return result

def may_export(card):
    return apply_license(card)['license_permissions']['modify']

def catalog_metadata():
    import csv
    path=ROOT/'data/manifest_v2.csv'
    if not path.exists():return {}
    with path.open(newline='') as handle:
        return {row['clip_id']:{'license':row['license'],'source':row['source'],
            'media_modified':row.get('media_modified',False),'manifest_entry':row} for row in csv.DictReader(handle)}

def resolve_cached_path(value):
    path=Path(value).resolve()
    if path.is_file():return path
    try:relative=path.relative_to(ROOT/'outputs')
    except ValueError:return path
    archived=ROOT/'outputs/retired_public'/relative
    return archived if archived.is_file() else path
