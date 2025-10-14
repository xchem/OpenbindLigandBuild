# Load the dependencies
import argparse
import pandas as pd
import pathlib
import yaml
import subprocess
import gemmi
import numpy as np


SCRIPT = """#/bin/sh
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
PDB="$DATASET_DIR"/{dtag}-pandda-input.pdb
MTZ="$DATASET_DIR"/{dtag}-pandda-input.mtz
MAP="$DATASET_DIR"/{event_map}
CIF={cif}
OUT="$MODELLED_STRUCTURES_DIR"/rhofit

mkdir "$OUT"
{pandda_2_dir}/scripts/pandda_rhofit.sh -pdb "$PDB" -map "$MAP" -mtz "$MTZ" -cif "$CIF" -out "$OUT"
cp "$MODELLED_STRUCTURES_DIR"/{dtag}-pandda-model.pdb "$MODELLED_STRUCTURES_DIR"/pandda-internal-fitted.pdb
cp "$OUT"/... "$MODELLED_STRUCTURES_DIR"/{dtag}-pandda-model.pdb

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

    # Submit
    
    # stdout, stderr = subprocess.Popen(
    #     f'chmod 777 {script_file}; sbatch {script_file}', 
    #     shell=True, 
    #     stdout=subprocess.PIPE, 
    #     stderr=subprocess.PIPE,
    #     )
    
def expand_event_map(bdc, ground_state_file, xmap_file, coord, out_file):
    ground_state_ccp4 = gemmi.read_ccp4_map(str(ground_state_file), setup=False)
    ground_state_ccp4.grid.spacegroup = gemmi.find_spacegroup_by_name('P1')
    ground_state_ccp4.setup(0.0)
    ground_state = ground_state_ccp4.grid 

    xmap_ccp4 = gemmi.read_ccp4_map(str(xmap_file), setup=False)
    xmap_ccp4.grid.spacegroup = gemmi.find_spacegroup_by_name('P1')
    xmap_ccp4.setup(0.0)
    xmap = xmap_ccp4.grid 

    mask = gemmi.FloatGrid(xmap.nu, xmap.nv, xmap.nw)
    mask.set_points_around(gemmi.Position(coord[0], coord[1], coord[2]), radius=10.0, value=1.0)

    event_map = gemmi.FloatGrid(xmap.nu, xmap.nv, xmap.nw)
    event_map.unit_cell = xmap.unit_cell
    event_map_array = np.array(event_map, copy=False)
    event_map_array[:,:,:] = np.array(xmap)[:,:,:] - (bdc*np.array(ground_state)[:,:,:])
    event_map_array[:,:,:] = event_map_array[:,:,:]*np.array(mask)[:,:,:]

    ccp4 = gemmi.Ccp4Map()
    ccp4.grid = event_map
    ccp4.update_ccp4_header()
    ccp4.write_ccp4_map(str(out_file))

def remove_nearby_atoms(pdb_file, coord, radius, output_file):
    st = gemmi.read_structure(str(pdb_file))
    new_st = st.clone()

    coord_array = np.array([coord[0], coord[1], coord[2]])

    # Delete chains
    chains_to_delete = []
    for model in st:
        for chain in model:
            chains_to_delete.append((model.num, chain.name))

    for model in new_st:
        for chain in model:
            for res in chain:
                print([model.num, chain.name])
                del chain[-1]
            print(len([x for x in chain]))

    # for model_name, chain_name in chains_to_delete:
    #     print(new_st)
    #     print([model_name, chain_name])
    #     del new_st[model_name][chain_name]

    # Add residues
    for model in st:
        for chain in model:
            # new_st.add_chain(gemmi.Chain(chain.name))
            for res in chain:
                add_res = True
                for atom in res:
                    pos = atom.pos
                    distance = np.linalg.norm(coord_array - np.array([pos.x, pos.y, pos.z]))
                    if distance < radius:
                        add_res = False
                
                if add_res:
                    new_st[model.num][chain.name].add_residue(res)
    
    new_st.write_pdb(str(output_file))


def main(pandda_dir):
    pandda_dir = pathlib.Path(pandda_dir)
    print('# PanDDA Dir')
    print(pandda_dir)

    # Determine which builds to perform. More than one binder is unlikely and score ranks well so build the best scoring event of each dataset.     
    panddas_events = pd.read_csv(pandda_dir / 'analyses' / 'pandda_analyse_events.csv')
    best_events = panddas_events[panddas_events['z_mean'] == panddas_events.groupby(by='dtag')['z_mean'].transform(max)]
    print('# Best Events')
    print(best_events)

    # Submit jobs
    print('# Jobs')
    for _, event_row in best_events.iterrows():
        dtag, event_idx, bdc, x, y, z = event_row['dtag'], event_row['event_idx'], event_row['1-BDC'], event_row['x'], event_row['y'], event_row['z']
        coord = [x,y,z]
        print(f'{dtag} : {event_idx}')
        dataset_dir = pandda_dir / 'processed_datasets' / dtag
        ligand_dir = dataset_dir / 'ligand_files'
        script_file = dataset_dir / f'rhofit_{event_idx}.slurm'
        ground_state_file = dataset_dir / GROUND_STATE_PATTERN.format(dtag=dtag)
        xmap_file = dataset_dir / 'xmap.ccp4'
        expanded_event_map = dataset_dir / 'event_map.ccp4'
        pdb_file = dataset_dir / f'{dtag}-pandda-input.pdb'
        restricted_pdb_file = dataset_dir / f'cut_input_model.pdb'

        print('# # Expand event map')
        expand_event_map(
            bdc,
            ground_state_file,
            xmap_file,
            coord,
            expanded_event_map,
        )

        remove_nearby_atoms(
            pdb_file,
            coord,
            10.0,
            restricted_pdb_file,
        )

        print('# # Script File')
        print(script_file)
        
        # Really all the cifs should be tried and the best used, or it should try the best cif from PanDDA
        # This is a temporary fix that will get 90% of situations that can be improved upon
        cifs = [x for x in ligand_dir.glob('*.cif')]
        if len(cifs) != 0:
            sbatch(
                SCRIPT.format(
                    pandda_dir=pandda_dir,
                    dtag=dtag,
                    event_map=expanded_event_map.name,
                    cif=cifs[0],
                    pandda_2_dir=PANDDA_2_DIR,
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