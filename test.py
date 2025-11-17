# Load the dependencies
import argparse
import pandas as pd
import pathlib
import subprocess
import gemmi
import numpy as np
import sys


SCRIPT = """#!/bin/sh
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=5120
#SBATCH --output=rhofit_{dtag}.o
#SBATCH --error=rhofit_{dtag}.e
#SBATCH --partition=cs05r

module load buster
# (probably module load pandda2 here)
            
DATASET_DIR="{pandda_dir}/processed_datasets/{dtag}"
MODELLED_STRUCTURES_DIR="$DATASET_DIR"/modelled_structures
PDB="$DATASET_DIR"/build.pdb
MTZ="$DATASET_DIR"/{dtag}-pandda-input.mtz
MAP="$DATASET_DIR"/{build_map}
CIF="$DATASET_DIR"/ligand_files/{cif}
OUT="$MODELLED_STRUCTURES_DIR"/rhofit

mkdir "$OUT"
{pandda_2_dir}/scripts/pandda_rhofit.sh -pdb "$PDB" -map "$MAP" -mtz "$MTZ" -cif "$CIF" -out "$OUT" -cut {cut}

# Make a copy of the previous structure model
cp "$MODELLED_STRUCTURES_DIR"/{dtag}-pandda-model.pdb "$MODELLED_STRUCTURES_DIR"/pandda-internal-fitted.pdb

# Merge new ligand into structure
{python} {merge_script} --dataset_dir="$DATASET_DIR"

"""

EVENT_MAP_PATTERN = '{dtag}-event_{event_idx}_1-BDC_{bdc}_map.native.ccp4'
GROUND_STATE_PATTERN = '{dtag}-ground-state-average-map.native.ccp4'

PANDDA_2_DIR = '/dls_sw/i04-1/software/PanDDA2'

def sbatch(script, script_file):
    print('# SCRIPT')
    print(script)

    # Write the script 
    with open(script_file, 'w') as f:
        f.write(script)

    # Submit - currently deactivated for testing
    
    # stdout, stderr = subprocess.Popen(
    #     f'chmod 777 {script_file}; sbatch {script_file}', 
    #     shell=True, 
    #     stdout=subprocess.PIPE, 
    #     stderr=subprocess.PIPE,
    #     )
    
def save_xmap(xmap, xmap_file):
    """Convenience script for saving ccp4 files."""
    ccp4 = gemmi.Ccp4Map()
    ccp4.grid = xmap
    ccp4.update_ccp4_header()
    ccp4.write_ccp4_map(str(xmap_file))

def read_pandda_map(xmap_file):
    """PanDDA 2 maps are often truncated, and PanDDA 1 maps can have misasigned spacegroups. 
    This method handles both."""
    dmap_ccp4 = gemmi.read_ccp4_map(str(xmap_file), setup=False)
    dmap_ccp4.grid.spacegroup = gemmi.find_spacegroup_by_name('P1')
    dmap_ccp4.setup(0.0)
    dmap = dmap_ccp4.grid 
    return dmap

def expand_event_map(bdc, ground_state_file, xmap_file, coord, out_file):
    """DEPRECATED. A method for recalculating event maps over the full cell."""
    ground_state_ccp4 = gemmi.read_ccp4_map(str(ground_state_file), setup=False)
    ground_state_ccp4.grid.spacegroup = gemmi.find_spacegroup_by_name('P1')
    ground_state_ccp4.setup(0.0)
    ground_state = ground_state_ccp4.grid 

    xmap_ccp4 = gemmi.read_ccp4_map(str(xmap_file), setup=False)
    xmap_ccp4.grid.spacegroup = gemmi.find_spacegroup_by_name('P1')
    xmap_ccp4.setup(0.0)
    xmap = xmap_ccp4.grid 

    mask = gemmi.FloatGrid(xmap.nu, xmap.nv, xmap.nw)
    mask.set_unit_cell(xmap.unit_cell)
    mask.set_points_around(gemmi.Position(coord[0], coord[1], coord[2]), radius=10.0, value=1.0)

    event_map = gemmi.FloatGrid(xmap.nu, xmap.nv, xmap.nw)
    event_map.set_unit_cell(xmap.unit_cell)
    event_map_array = np.array(event_map, copy=False)
    event_map_array[:,:,:] = np.array(xmap)[:,:,:] - (bdc*np.array(ground_state)[:,:,:])
    event_map_array[:,:,:] = event_map_array[:,:,:]*np.array(mask)[:,:,:]

    event_map_non_zero = event_map_array[event_map_array != 0.0]
    cut = np.std(event_map_non_zero)

    return cut

def mask_map(dmap, coord, radius=10.0):
    """Simple routine to mask density to region around a specified point."""
    mask = gemmi.FloatGrid(dmap.nu, dmap.nv, dmap.nw)
    mask.set_unit_cell(dmap.unit_cell)
    mask.set_points_around(gemmi.Position(coord[0], coord[1], coord[2]), radius=radius, value=1.0)

    dmap_array = np.array(dmap, copy=False)
    dmap_array[:,:,:] = dmap_array[:,:,:] * np.array(mask)[:,:,:]

    return dmap 


def remove_nearby_atoms(pdb_file, coord, radius, output_file):
    """An inelegant method for removing residues near the event centroid and creating
    a new, truncated pdb file. GEMMI doesn't have a super nice way to remove
    residues according to a specific criteria."""
    st = gemmi.read_structure(str(pdb_file))
    new_st = st.clone()  # Clone to keep metadata

    coord_array = np.array([coord[0], coord[1], coord[2]])

    # Delete all residues for a clean chain. Yes this is an arcane way to do it.
    chains_to_delete = []
    for model in st:
        for chain in model:
            chains_to_delete.append((model.num, chain.name))

    for model in new_st:
        for chain in model:
            for res in chain:
                del chain[-1]

    # Add non-rejected residues to a new structure
    for j, model in enumerate(st):
        for k, chain in enumerate(model):
            for res in chain:
                add_res = True
                for atom in res:
                    pos = atom.pos
                    distance = np.linalg.norm(coord_array - np.array([pos.x, pos.y, pos.z]))
                    if distance < radius:
                        add_res = False
                
                if add_res:
                    new_st[j][k].add_residue(res)
                else:
                    print(f'Dropped residue: {j} {k} {res.name}')
    new_st.write_pdb(str(output_file))


def main(pandda_dir):
    pandda_dir = pathlib.Path(pandda_dir)
    print('# PanDDA Dir')
    print(pandda_dir)

    # Determine which builds to perform. More than one binder is unlikely and score ranks 
    # well so build the best scoring event of each dataset.     
    panddas_events = pd.read_csv(pandda_dir / 'analyses' / 'pandda_analyse_events.csv')
    best_events = panddas_events[panddas_events['z_mean'] == panddas_events.groupby(by='dtag')['z_mean'].transform(max)]
    print('# Best Events')
    print(best_events)

    # Submit jobs
    print('# Jobs')
    for _, event_row in best_events.iterrows():
        dtag, event_idx, bdc, x, y, z = (event_row['dtag'], 
                                         event_row['event_idx'], 
                                         event_row['1-BDC'], 
                                         event_row['x'], 
                                         event_row['y'], 
                                         event_row['z'],
        )
        coord = [x,y,z]
        print(f'{dtag} : {event_idx}')
        dataset_dir = (pandda_dir / 'processed_datasets' / dtag).resolve()
        ligand_dir = dataset_dir / 'ligand_files'
        script_file = dataset_dir / f'rhofit_{event_idx}.slurm'
        build_dmap = dataset_dir / f'{dtag}-z_map.native.ccp4'
        restricted_build_dmap = dataset_dir / 'build.ccp4'
        pdb_file = dataset_dir / f'{dtag}-pandda-input.pdb'
        restricted_pdb_file = dataset_dir / 'build.pdb'
        python = sys.executable
        merge_script = pathlib.Path(python).parent.parent.parent / 'merge.py'
        dmap_cut = 2.0  # This is usually quite a good contour for building and consistent 
                        # (usually) with the cutoffs PanDDA 2 uses for event finding

        # Rhofit can be confused by hunting non-binding site density. This can be avoided
        # by truncating the map to near the binding site
        dmap = read_pandda_map(build_dmap)
        dmap = mask_map(dmap, coord)
        save_xmap(dmap, restricted_build_dmap)

        # Rhofit masks the protein before building. If the original protein 
        # model clips the event then this results in autobuilding becoming impossible.
        # To address tis residues within a 10A neighbourhood of the binding event
        # are removed.
        print('# # Remove nearby atoms to make room for autobuilding')
        remove_nearby_atoms(
            pdb_file,
            coord,
            10.0,
            restricted_pdb_file,
        )

        print('# # Script File')
        print(script_file)
        
        # Really all the cifs should be tried and the best used, or it should try the best 
        # cif from PanDDA
        # This is a temporary fix that will get 90% of situations that can be improved upon
        cifs = [x for x in ligand_dir.glob('*.cif')]
        if len(cifs) != 0:
            sbatch(
                SCRIPT.format(
                    pandda_dir=pandda_dir.resolve(),
                    dtag=dtag,
                    build_map=build_dmap.name,
                    cif=cifs[0].name,
                    pandda_2_dir=PANDDA_2_DIR,
                    cut=dmap_cut,
                    python=python,
                    merge_script= merge_script,
                ),
                script_file
            )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
                    prog='ProgramName',
                    description='What the program does',
                    epilog='Text at the bottom of help')
    
    parser.add_argument('--pandda_dir')
    args = parser.parse_args()
    print(args)

    main(
        args.pandda_dir,
    )