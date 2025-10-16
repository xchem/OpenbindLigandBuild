# Load the dependencies
import argparse
import pandas as pd
import pathlib
import yaml
import subprocess
import gemmi
import numpy as np

PROTEIN_RESIDUES = ["ALA",
                 "ARG",
                 "ASN",
                 "ASP",
                 "CYS",
                 "GLN",
                 "GLU",
                 "HIS",
                 "ILE",
                 "LEU",
                 "LYS",
                 "MET",
                 "PHE",
                 "PRO",
                 "SER",
                 "THR",
                 "TRP",
                 "TYR",
                 "VAL",
                 "GLY",
                 ]

def get_contact_chain(protein_st, ligand_st):
    
    ligand_pos_list = []
    for model in protein_st:
        for chain in model:
            for res in chain:
                for atom in res:
                    pos = atom.pos
                    ligand_pos_list.append([pos.x, pos.y, pos.z])
    centroid = np.linalg.norm(np.array(ligand_pos_list), axis=0)
    
    chain_counts = {}
    for model in protein_st:
        for chain in model:
            chain_counts[chain.name] = 0
            for res in chain:
                if res.name not in PROTEIN_RESIDUES:
                    continue
                for atom in res:
                    pos = atom.pos
                    distance = np.linalg.norm(np.array([pos.x, pos.y, pos.z]) - centroid)
                    if distance < 5.0:
                        chain_counts[chain.name] += 1

    return min(chain_counts, key = lambda _x: chain_counts[_x])


def main(dataset_dir):
    """Basic routine is to get the initial protein structure, the ligand structure and
      add the ligand to the protein chain with the most contacts in a few A."""
    dataset_dir = pathlib.Path(dataset_dir)
    dtag = dataset_dir.name
    protein_st_file = dataset_dir / f'{dtag}-pandda-input.pdb'
    ligand_st_file = dataset_dir / 'modelled_structures' / 'rhofit' / 'rhofit' / 'best.pdb'

    protein_st = gemmi.read_structure(str(protein_st_file))
    ligand_st = gemmi.read_structure(str(ligand_st_file))

    contact_chain = get_contact_chain(protein_st, ligand_st)
    print(contact_chain)
    protein_st[0][contact_chain].add_residue(ligand_st[0][0][0])



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
                    prog='ProgramName',
                    description='What the program does',
                    epilog='Text at the bottom of help')
    
    parser.add_argument('--dataset_dir')
    args = parser.parse_args()
    print(args)

    main(
        args.dataset_dir,
    )