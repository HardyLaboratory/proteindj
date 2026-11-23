"""Tier A unit tests for align_boltz.py.

align_boltz.py runs `setup_logging()` at import time, which creates an
"alignment.log" FileHandler in the current working directory as a side
effect. We import it with cwd temporarily pointed at a throwaway directory
so this doesn't litter the repo (this only matters on the first import in
the whole test session, since module imports are cached).
"""
import json
import os
import tempfile
from pathlib import Path

import pytest
from Bio.PDB import PDBParser

_prev_cwd = os.getcwd()
os.chdir(tempfile.mkdtemp())
try:
    from align_boltz import (
        get_all_ca_atoms, get_chain_ca_atoms, get_target_ca_atoms,
        get_designed_ca_indices, calculate_designed_metrics, align_structures,
    )
finally:
    os.chdir(_prev_cwd)


def _write_pdb(tmp_path, lines, name="input.pdb"):
    path = tmp_path / name
    path.write_text("\n".join(lines) + "\n")
    return path


def _parse(path):
    parser = PDBParser(QUIET=True)
    return parser.get_structure('s', path)


def _multi_chain_structure(tmp_path, pdb_atom_line):
    lines = (
        [pdb_atom_line(serial=1, name='CA', chain='A', resnum=1)]
        + [pdb_atom_line(serial=2, name='CA', chain='B', resnum=1, x=20.0)]
        + [pdb_atom_line(serial=3, name='CA', chain='C', resnum=1, x=40.0)]
    )
    return _parse(_write_pdb(tmp_path, lines))


# ---------------------------------------------------------------------------
# get_all_ca_atoms
# ---------------------------------------------------------------------------

def test_get_all_ca_atoms_collects_across_chains(tmp_path, pdb_atom_line):
    structure = _multi_chain_structure(tmp_path, pdb_atom_line)
    atoms = get_all_ca_atoms(structure)
    assert len(atoms) == 3


def test_get_all_ca_atoms_no_ca_raises(tmp_path, pdb_atom_line):
    lines = [pdb_atom_line(serial=1, name='N', chain='A', resnum=1)]
    structure = _parse(_write_pdb(tmp_path, lines))
    with pytest.raises(ValueError, match="No CA atoms found in structure"):
        get_all_ca_atoms(structure)


# ---------------------------------------------------------------------------
# get_chain_ca_atoms
# ---------------------------------------------------------------------------

def test_get_chain_ca_atoms_filters_by_chain(tmp_path, pdb_atom_line):
    structure = _multi_chain_structure(tmp_path, pdb_atom_line)
    atoms = get_chain_ca_atoms(structure, 'B')
    assert len(atoms) == 1


def test_get_chain_ca_atoms_missing_chain_raises(tmp_path, pdb_atom_line):
    structure = _multi_chain_structure(tmp_path, pdb_atom_line)
    with pytest.raises(ValueError, match="No CA atoms found in chain Z"):
        get_chain_ca_atoms(structure, 'Z')


# ---------------------------------------------------------------------------
# get_target_ca_atoms
# ---------------------------------------------------------------------------

def test_get_target_ca_atoms_excludes_binder_chain_handles_split_targets(tmp_path, pdb_atom_line):
    structure = _multi_chain_structure(tmp_path, pdb_atom_line)
    atoms = get_target_ca_atoms(structure, binder_chain='A')
    assert len(atoms) == 2


def test_get_target_ca_atoms_no_target_chains_raises(tmp_path, pdb_atom_line):
    lines = [pdb_atom_line(serial=1, name='CA', chain='A', resnum=1)]
    structure = _parse(_write_pdb(tmp_path, lines))
    with pytest.raises(ValueError, match="No CA atoms found in target chains"):
        get_target_ca_atoms(structure, binder_chain='A')


# ---------------------------------------------------------------------------
# designed residue metrics
# ---------------------------------------------------------------------------

def _monomer_pdb(tmp_path, pdb_atom_line, name, xs, bfactors=None):
    bfactors = bfactors or [0.0] * len(xs)
    lines = [pdb_atom_line(serial=i + 1, name='CA', chain='A', resnum=i + 1, x=x, y=(i % 2) * 1.0, temp=b)
             for i, (x, b) in enumerate(zip(xs, bfactors))]
    return _write_pdb(tmp_path, lines, name=name)


def test_get_designed_ca_indices_selects_unfixed_residues(tmp_path, pdb_atom_line):
    structure = _parse(_monomer_pdb(tmp_path, pdb_atom_line, "d.pdb", [0, 3, 6, 9]))
    assert get_designed_ca_indices(structure, [True, False, False, True]) == [1, 2]
    assert get_designed_ca_indices(structure, [True] * 4) == []


def test_get_designed_ca_indices_length_mismatch_returns_none(tmp_path, pdb_atom_line):
    structure = _parse(_monomer_pdb(tmp_path, pdb_atom_line, "d.pdb", [0, 3, 6, 9]))
    assert get_designed_ca_indices(structure, [True, False]) is None


def test_calculate_designed_metrics_uses_designed_residues_only(tmp_path, pdb_atom_line):
    ref = get_all_ca_atoms(_parse(_monomer_pdb(tmp_path, pdb_atom_line, "r.pdb", [0, 3, 6, 9])))
    pred = get_all_ca_atoms(_parse(_monomer_pdb(
        tmp_path, pdb_atom_line, "p.pdb", [0, 3, 6, 9], bfactors=[90.0, 80.0, 60.0, 90.0])))
    # Move predicted residue 2 by 2 A in x so only it deviates
    pred[2].set_coord(pred[2].coord + [2.0, 0, 0])

    metrics = calculate_designed_metrics(ref, pred, [1, 2])
    assert metrics == {"boltz_rmsd_designed": 1.41, "boltz_plddt_designed": 0.7}
    assert calculate_designed_metrics(ref, pred, []) == {}


def _align_task(tmp_path, pdb_atom_line, fixed_mask):
    design = _monomer_pdb(tmp_path, pdb_atom_line, "fold_0_seq_0.pdb", [0, 3, 6, 9])
    boltz = _monomer_pdb(tmp_path, pdb_atom_line, "fold_0_seq_0_boltzpred.pdb", [0, 3, 6, 9],
                         bfactors=[90.0, 80.0, 60.0, 90.0])
    src_json = tmp_path / "src.json"
    src_json.write_text(json.dumps({"confidence_score": 0.9, "ptm": 0.9, "complex_plddt": 0.8, "complex_pde": 1.0}))
    dst_json = tmp_path / "dst.json"
    result = align_structures((design, boltz, tmp_path / "out.pdb", src_json, dst_json,
                               0, 0, "monomer", fixed_mask))
    return result, json.loads(dst_json.read_text())


def test_align_structures_monomer_reports_designed_metrics(tmp_path, pdb_atom_line):
    result, out = _align_task(tmp_path, pdb_atom_line, [True, False, False, True])
    assert result[2] is None
    assert out["boltz_rmsd_designed"] == 0.0
    assert out["boltz_plddt_designed"] == 0.7
    assert out["boltz_rmsd_overall"] == 0.0


def test_align_structures_monomer_without_mask_omits_designed_metrics(tmp_path, pdb_atom_line):
    _, out = _align_task(tmp_path, pdb_atom_line, None)
    assert "boltz_rmsd_designed" not in out
    assert "boltz_plddt_designed" not in out
