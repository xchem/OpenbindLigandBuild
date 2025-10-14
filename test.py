# Load the dependencies
import argparse
import pandas as pd
import pathlib
import yaml
import subprocess

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
MODELLED_STRUCTURES_DIR=DATASET_DIR/modelled_structures
PDB=DATASET_DIR/"{dtag}-pandda-input.pdb"
MTZ=DATASET_DIR/"{dtag}-pandda-input.mtz"
MAP=DATASET_DIR/"{event_map}"
CIF=DATASET_DIR/ligand_files/"{cif}"
OUT=MODELLED_STRUCTURES_DIR/rhofit

PanDDA2/scripts/pandda_rhofit.sh -pdb $PDB -map $MAP -mtz $MTZ -cif $CIF -out $OUT
cp MODELLED_STRUCTURES_DIR/pandda-model.pdb pandda-internal-fitted.pdb
cp OUT/... MODELLED_STRUCTURES_DIR/pandda-model.pdb

"""

EVENT_MAP_PATTERN = '{dtag}-event_{event_idx}_1-BDC_{bdc}_map.native.ccp4'

def sbatch(script, script_file):
    print(script)

    # Write the script 
    with open(script_file, 'w') as f:
        f.write(script)

    # Submit
    stdout, stderr = subprocess.Popen(
        f'chmod 777 {script_file}; sbatch {script_file}', 
        shell=True, 
        stdout=subprocess.PIPE, 
        stderr=subprocess.PIPE,
        )

def main(pandda_dir):
    pandda_dir = pathlib.Path(pandda_dir)
    print(pandda_dir)

    # Determine which builds to perform. More than one binder is unlikely and score ranks well so build the best scoring event of each dataset.     
    panddas_events = pd.read_csv(pandda_dir / 'analyses' / 'pandda_analyse_events.csv')
    best_events = panddas_events[panddas_events['z_mean'] == panddas_events.groupby(by='dtag')['z_mean'].transform(max)]
    print(best_events)

    # Submit jobs
    for _, event_row in best_events.iterrows():
        dtag, event_idx, bdc = event_row['dtag'], event_row['event_idx'], event_row['1-BDC']
        print(f'{dtag} : {event_idx}')
        dataset_dir = pandda_dir / 'processed_datasets' / dtag
        ligand_dir = dataset_dir / 'ligand_files'
        
        # Really all the cifs should be tried and the best used, or it should try the best cif from PanDDA
        # This is a temporary fix that will get 90% of situations that can be improved upon
        cifs = [x for x in ligand_dir.glob('*.cif')]
        if len(cifs) != 0:
            sbatch(
                SCRIPT.format(
                    pandda_dir=pandda_dir,
                    dtag=dtag,
                    event_map=EVENT_MAP_PATTERN.format(dtag=dtag, event_idx=event_idx, bdc=bdc),
                    cif=cifs[0],
                ),
                dataset_dir / f'rhofit_{event_idx}.slurm'
            )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
                    prog='ProgramName',
                    description='What the program does',
                    epilog='Text at the bottom of help')
    
    parser.add_argument('--pandda_dir')
    parser.parse_args()

    main(
        parser[0],
    )