from pathlib import Path
import subprocess

from PIL import Image

from truetake.presentation import plain_reason
from truetake.thumbnails import midpoint_thumbnail


def test_midpoint_thumbnail_is_real_320px_image(tmp_path):
    video = tmp_path / 'clip.mp4'
    subprocess.run(['ffmpeg', '-v', 'error', '-f', 'lavfi', '-i',
                    'color=c=white:s=96x64:r=5:d=2', '-c:v', 'libx264', '-threads', '1', str(video)], check=True)
    thumbnail = midpoint_thumbnail(str(video), str(tmp_path / 'midpoint.jpg'), 2.)
    with Image.open(thumbnail) as image:
        assert image.width == 320
        assert image.convert('L').getextrema()[0] > 200


def test_short_clip_reason_does_not_claim_a_missing_goal():
    card = {'duration_s': 2.8, 'verdict': 'NEEDS REVIEW', 'decision_reasons': ['Goal is not confirmed']}
    assert plain_reason(card) == 'Too short to review (2.8 s)'
    card.update(duration_s=8., measured={'usable_assessment': {'target': 10.}})
    assert plain_reason(card) == 'Too short to review (8.0 s)'


def test_example_uses_one_hand_and_plain_idle_text():
    from truetake.contracts import load_contract
    for name in ['mercor_folding', 'wash_dishes']:
        contract = load_contract(f'contracts/{name}.yaml')
        rules = {r['id']: r for r in contract['process']}
        assert rules['one_hand_visible']['measurement'] == 'hand_present'
        assert rules['idle']['description'] == 'No idle stretch over 3 s'
