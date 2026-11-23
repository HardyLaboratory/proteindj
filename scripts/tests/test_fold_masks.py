"""Tests for fold_masks.py."""
import json

from fold_masks import find_fold_json, load_fixed_mask


def test_find_fold_json_matches_fold_prefix(tmp_path):
    (tmp_path / 'fold_3.json').write_text('{}')
    assert find_fold_json(str(tmp_path), 'fold_3_seq_12.pdb') == str(tmp_path / 'fold_3.json')
    assert find_fold_json(str(tmp_path), 'fold_4_seq_0.pdb') is None
    assert find_fold_json(str(tmp_path), 'other.pdb') is None


def test_load_fixed_mask_rfd_requires_fixed_sequence_and_structure(tmp_path):
    path = tmp_path / 'fold_0.json'
    path.write_text(json.dumps({
        'rfd_inpaint_seq': [True, True, False, False],
        'rfd_inpaint_str': [True, False, True, False],
    }))
    # Residue 1 has its sequence provided but its structure was diffused, so it is designed
    assert load_fixed_mask(str(path)) == [True, False, False, False]


def test_load_fixed_mask_other_tools_use_sequence_mask(tmp_path):
    path = tmp_path / 'fold_0.json'
    path.write_text(json.dumps({'bg_inpaint_seq': [True, False, True]}))
    assert load_fixed_mask(str(path)) == [True, False, True]


def test_load_fixed_mask_returns_none_without_mask(tmp_path):
    path = tmp_path / 'fold_0.json'
    path.write_text(json.dumps({'fold_id': 0}))
    assert load_fixed_mask(str(path)) is None
