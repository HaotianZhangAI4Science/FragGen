import subprocess
from glob import glob 
import os.path as osp
import time
import numpy as np
#targets = glob('./data/alphafold/resgen_epo187/*')
#out_dir = '/home/haotian/Molecule_Generation/ResGen-main/data/alphafold/resgen_new'
targets = np.sort(glob('./data/crosssdock_test/*'))
out_dir = './SurfFrag_cartesian'
config_file = './configs/sample_cartesian.yml'

for target in targets:
    start_time = time.time()
    pdb_file = glob(osp.join(target, '*.pdb'))[0]
    lig_file = glob(osp.join(target, '*.sdf'))[0]
    surface_file = glob(osp.join(target, '*.ply'))[0]

    gen_sdfs =  osp.join(out_dir,target.split('/')[-1], 'SDF')
    gen_sdfs = glob(osp.join(gen_sdfs,'*'))
    if len(gen_sdfs) > 0:
        continue
    command = f'python gen_from_pdb.py --config {config_file} --surf_file {surface_file} --pdb_file {pdb_file} --sdf_file {lig_file} --save_dir {out_dir}'
    print(command)
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    # Check if fpocket command was successful
    if result.returncode == 0:
        print('executed successfully.')
        print('Output:')
        print(result.stdout)
        print('consumed time: ',time.time()-start_time)
    else:
        print('execution failed.')
        print('Error:')
        print(result.stderr)


# for target in targets:
#     start_time = time.time()
#     pdb_file = glob(osp.join(target, '*.pdb'))[0]
#     lig_file = glob(osp.join(target, '*.sdf'))[0]
#     command = f'python gen.py --pdb_file {pdb_file} --sdf_file {lig_file} --outdir {out_dir}'

#     result = subprocess.run(command, shell=True, capture_output=True, text=True)

#     # Check if fpocket command was successful
#     if result.returncode == 0:
#         print('executed successfully.')
#         print('Output:')
#         print(result.stdout)
#         print('consumed time: ',time.time()-start_time)
#     else:
#         print('execution failed.')
#         print('Error:')
#         print(result.stderr)
