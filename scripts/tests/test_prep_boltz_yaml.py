"""Tier A unit tests for prep_boltz_yaml.py.

PPBuilder (Bio.PDB) determines peptide-bond continuity purely from the
C(i)--N(i+1) atom distance vs `radius` (aliased here as max_break_distance),
so synthetic fixtures only need N/C atoms at controlled distances - full
realistic backbone geometry isn't required.
"""
import json

import yaml
from Bio.PDB import PDBParser

from prep_boltz_yaml import (
    get_available_chain_ids,
    split_structure_chains,
    add_seqres_to_pdb,
    generate_yaml_config,
    get_chain_ids,
    process_pdb_files,
)


def _res_atom_lines(pdb_atom_line, chain, resnum, n_coord, c_coord, resname='ALA', serial_start=1):
    return [
        pdb_atom_line(serial=serial_start, name='N', chain=chain, resnum=resnum,
                       resname=resname, x=n_coord[0], y=n_coord[1], z=n_coord[2]),
        pdb_atom_line(serial=serial_start + 1, name='C', chain=chain, resnum=resnum,
                       resname=resname, x=c_coord[0], y=c_coord[1], z=c_coord[2]),
    ]


def _write_broken_chain_pdb(tmp_path, pdb_atom_line):
    """Chain A: residues 1-2 bonded (C-N dist 0.3), break to 3-4 (C-N dist 8.0), bonded again."""
    lines = []
    lines += _res_atom_lines(pdb_atom_line, 'A', 1, (0, 0, 0), (1, 0, 0), serial_start=1)
    lines += _res_atom_lines(pdb_atom_line, 'A', 2, (1.3, 0, 0), (2, 0, 0), serial_start=3)
    lines += _res_atom_lines(pdb_atom_line, 'A', 3, (10, 0, 0), (11, 0, 0), serial_start=5)
    lines += _res_atom_lines(pdb_atom_line, 'A', 4, (11.3, 0, 0), (12, 0, 0), serial_start=7)
    path = tmp_path / "input.pdb"
    path.write_text("\n".join(lines) + "\n")
    return path


def _parse(path):
    parser = PDBParser(QUIET=True)
    return parser.get_structure('s', path)


# ---------------------------------------------------------------------------
# get_available_chain_ids
# ---------------------------------------------------------------------------

def test_get_available_chain_ids_excludes_used():
    result = get_available_chain_ids({'A', 'B'})
    assert result[:3] == ['C', 'D', 'E']
    assert 'A' not in result
    assert 'B' not in result


# ---------------------------------------------------------------------------
# split_structure_chains
# ---------------------------------------------------------------------------

def test_split_structure_chains_splits_on_break(tmp_path, pdb_atom_line):
    structure = _parse(_write_broken_chain_pdb(tmp_path, pdb_atom_line))
    new_structure, chain_id_map = split_structure_chains(structure, max_break_distance=3.0)

    assert chain_id_map['A'] == ['A', 'B']
    new_chain_ids = sorted(chain.id for chain in new_structure[0])
    assert new_chain_ids == ['A', 'B']

    chain_a_residues = [r.id[1] for r in new_structure[0]['A']]
    chain_b_residues = [r.id[1] for r in new_structure[0]['B']]
    assert chain_a_residues == [1, 2]
    assert chain_b_residues == [3, 4]


def test_split_structure_chains_no_break_keeps_single_chain(tmp_path, pdb_atom_line):
    lines = []
    lines += _res_atom_lines(pdb_atom_line, 'A', 1, (0, 0, 0), (1, 0, 0), serial_start=1)
    lines += _res_atom_lines(pdb_atom_line, 'A', 2, (1.3, 0, 0), (2, 0, 0), serial_start=3)
    path = tmp_path / "input.pdb"
    path.write_text("\n".join(lines) + "\n")
    structure = _parse(path)

    new_structure, chain_id_map = split_structure_chains(structure, max_break_distance=3.0)
    assert chain_id_map == {'A': ['A']}
    assert [chain.id for chain in new_structure[0]] == ['A']


# ---------------------------------------------------------------------------
# add_seqres_to_pdb
# ---------------------------------------------------------------------------

def test_add_seqres_to_pdb_writes_seqres_per_split_chain(tmp_path, pdb_atom_line):
    input_path = _write_broken_chain_pdb(tmp_path, pdb_atom_line)
    output_path = tmp_path / "with_seqres.pdb"
    add_seqres_to_pdb(input_path, output_path, max_break_distance=3.0)

    text = output_path.read_text()
    seqres_lines = [l for l in text.splitlines() if l.startswith('SEQRES')]
    assert any(l.startswith('SEQRES   1 A    2  ALA ALA') for l in seqres_lines)
    assert any(l.startswith('SEQRES   1 B    2  ALA ALA') for l in seqres_lines)
    assert text.rstrip().splitlines()[-1] == 'END'


# ---------------------------------------------------------------------------
# add_seqres_to_pdb / process_pdb_files with a fixed-residue mask
# ---------------------------------------------------------------------------

def _atom_resnums(pdb_path):
    return [(l[21], int(l[22:26])) for l in pdb_path.read_text().splitlines()
            if l.startswith('ATOM') and l[12:16].strip() == 'N']


def _write_bonded_chain_pdb(path, pdb_atom_line, n_residues=4, chain='A'):
    lines = []
    for i in range(n_residues):
        lines += _res_atom_lines(pdb_atom_line, chain, i + 1, (i * 1.3, 0, 0),
                                 (i * 1.3 + 1, 0, 0), serial_start=2 * i + 1)
    path.write_text("\n".join(lines) + "\n")
    return path


def test_add_seqres_to_pdb_fixed_mask_drops_designed_atoms_but_keeps_seqres(tmp_path, pdb_atom_line):
    input_path = _write_bonded_chain_pdb(tmp_path / 'in.pdb', pdb_atom_line)
    output_path = tmp_path / 'out.pdb'
    chains = add_seqres_to_pdb(input_path, output_path, fixed_mask=[True, False, False, True])

    assert chains == ['A']
    assert _atom_resnums(output_path) == [('A', 1), ('A', 4)]
    assert any(l.startswith('SEQRES   1 A    4  ALA ALA ALA ALA')
               for l in output_path.read_text().splitlines())


def test_add_seqres_to_pdb_fixed_mask_detects_breaks_on_full_structure(tmp_path, pdb_atom_line):
    input_path = _write_broken_chain_pdb(tmp_path, pdb_atom_line)
    output_path = tmp_path / 'out.pdb'
    # Removing residue 2 must not change where the chain is split (between residues 2 and 3)
    chains = add_seqres_to_pdb(input_path, output_path, fixed_mask=[True, False, True, True])

    assert chains == ['A', 'B']
    assert _atom_resnums(output_path) == [('A', 1), ('B', 3), ('B', 4)]
    seqres = [l for l in output_path.read_text().splitlines() if l.startswith('SEQRES')]
    assert any(l.startswith('SEQRES   1 A    2') for l in seqres)
    assert any(l.startswith('SEQRES   1 B    2') for l in seqres)


def test_add_seqres_to_pdb_fixed_mask_omits_fully_designed_chains(tmp_path, pdb_atom_line):
    input_path = _write_broken_chain_pdb(tmp_path, pdb_atom_line)
    output_path = tmp_path / 'out.pdb'
    chains = add_seqres_to_pdb(input_path, output_path, fixed_mask=[False, False, True, True])

    assert chains == ['B']
    assert not any(l.startswith('SEQRES') and ' A ' in l for l in output_path.read_text().splitlines())


def test_add_seqres_to_pdb_mismatched_mask_templates_everything(tmp_path, pdb_atom_line):
    input_path = _write_bonded_chain_pdb(tmp_path / 'in.pdb', pdb_atom_line)
    output_path = tmp_path / 'out.pdb'
    add_seqres_to_pdb(input_path, output_path, fixed_mask=[True, False])

    assert len(_atom_resnums(output_path)) == 4


def _run_process(tmp_path, pdb_atom_line, mask):
    in_dir = tmp_path / 'in'
    json_dir = tmp_path / 'jsons'
    in_dir.mkdir()
    json_dir.mkdir()
    _write_bonded_chain_pdb(in_dir / 'fold_0_seq_0.pdb', pdb_atom_line)
    (json_dir / 'fold_0.json').write_text(json.dumps({'rfd_inpaint_seq': mask}))
    out_dir = tmp_path / 'out'
    process_pdb_files(str(in_dir), str(out_dir), use_template=True, template_chain='A',
                      fold_json_dir=str(json_dir))
    return out_dir, yaml.safe_load((out_dir / 'fold_0_seq_0.yaml').read_text())


def test_process_pdb_files_templates_only_fixed_residues(tmp_path, pdb_atom_line):
    out_dir, config = _run_process(tmp_path, pdb_atom_line, [True, True, False, False])

    assert config['templates'] == [{'pdb': 'templates/fold_0_seq_0.pdb', 'chain_id': 'A'}]
    assert _atom_resnums(out_dir / 'templates' / 'fold_0_seq_0.pdb') == [('A', 1), ('A', 2)]


def test_process_pdb_files_skips_template_when_all_residues_designed(tmp_path, pdb_atom_line):
    out_dir, config = _run_process(tmp_path, pdb_atom_line, [False] * 4)

    assert 'templates' not in config
    assert not (out_dir / 'templates' / 'fold_0_seq_0.pdb').exists()


# ---------------------------------------------------------------------------
# get_chain_ids
# ---------------------------------------------------------------------------

def test_get_chain_ids_extracts_ids():
    sequences = [{'id': 'A', 'sequence': 'AG', 'msa': 'empty'},
                 {'id': 'C', 'sequence': 'GG', 'msa': 'empty'}]
    assert get_chain_ids(sequences) == ['A', 'C']


# ---------------------------------------------------------------------------
# generate_yaml_config
# ---------------------------------------------------------------------------

def test_generate_yaml_config_without_template():
    sequences = [{'id': 'A', 'sequence': 'AG', 'msa': 'empty'}]
    config = generate_yaml_config(sequences)
    assert config == {'sequences': [{'protein': {'id': 'A', 'sequence': 'AG', 'msa': 'empty'}}]}
    assert 'templates' not in config


def test_generate_yaml_config_with_template():
    sequences = [{'id': 'A', 'sequence': 'AG', 'msa': 'empty'}]
    config = generate_yaml_config(
        sequences, use_template=True, pdb_filename='fold_1.pdb',
        template_chains=['B'], template_force=True, template_threshold=2.5,
    )
    assert config['templates'] == [
        {'pdb': 'templates/fold_1.pdb', 'chain_id': 'B', 'force': True, 'threshold': 2.5}
    ]


def test_generate_yaml_config_template_requires_all_fields():
    sequences = [{'id': 'A', 'sequence': 'AG', 'msa': 'empty'}]
    # use_template True but no pdb_filename/template_chains -> no templates block
    config = generate_yaml_config(sequences, use_template=True)
    assert 'templates' not in config
