"""Helpers for reading the per-residue designed/fixed masks recorded by fold design."""
import os
import re
import json

# Fold-design JSON keys holding the per-residue sequence-fixed mask (True = fixed, False = designed).
# Only one is present, depending on the fold design tool (RFdiffusion, BindCraft, BoltzGen).
INPAINT_SEQ_KEYS = ('rfd_inpaint_seq', 'bc_inpaint_seq', 'bg_inpaint_seq')


def find_fold_json(fold_json_dir, pdb_filename):
    """Return the path of the fold design JSON (fold_N.json) matching fold_N_seq_M*.pdb, or None."""
    match = re.match(r'(fold_\d+)_seq_\d+', os.path.basename(pdb_filename))
    if not match:
        return None
    json_path = os.path.join(fold_json_dir, f'{match.group(1)}.json')
    return json_path if os.path.isfile(json_path) else None


def load_fixed_mask(json_path):
    """
    Load the per-residue mask of residues whose sequence was held fixed during fold design
    (True = fixed, False = designed), ordered as the residues of the design PDB.

    For RFdiffusion, residues whose structure was diffused (rfd_inpaint_str False) are also
    treated as designed, even if their sequence was provided.

    Returns None if the JSON has no usable mask.
    """
    with open(json_path) as f:
        data = json.load(f)
    key = next((k for k in INPAINT_SEQ_KEYS if isinstance(data.get(k), list)), None)
    if key is None:
        return None
    mask = [bool(v) for v in data[key]]
    inpaint_str = data.get('rfd_inpaint_str')
    if key == 'rfd_inpaint_seq' and isinstance(inpaint_str, list) and len(inpaint_str) == len(mask):
        mask = [seq_fixed and bool(str_fixed) for seq_fixed, str_fixed in zip(mask, inpaint_str)]
    return mask
