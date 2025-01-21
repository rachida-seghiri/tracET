import nrrd
from tracET.core.diff import prepare_input
from tracET.core.skel import surface_skel
from pathlib import Path
import numpy as np
import mrcfile
import matplotlib.pyplot as plt

data = Path('/mnt/big_data/rseghiri/nnUNet_datasets/predictions_9')
target_dir = Path('/mnt/big_data/rseghiri/skeletons/predictions_9')
source_file_format = 'nrrd'


print(f"Creating target directory {str(target_dir)}")
print(f"Source file format: {source_file_format}")
target_dir.mkdir(parents=True, exist_ok=True)

all_tomos = list(data.glob(f"*.{source_file_format}"))

def load_mrc(fname, mmap=False, no_saxes=True):
    """
    Load an input MRC tomogram as ndarray

    :param fname: the input MRC
    :param mmap: if True (default False) the data are read as a memory map
    :param no_saxes: if True (default) then X and Y axes are swaped to cancel the swaping made by mrcfile package
    :return: a ndarray (or memmap is mmap=True)
    """
    if mmap:
        mrc = mrcfile.mmap(fname, permissive=True, mode='r+')
    else:
        mrc = mrcfile.open(fname, permissive=True, mode='r+')
    if no_saxes:
        return np.swapaxes(mrc.data, 0, 2)
    return mrc.data

def get_save_path(target_dir: Path, stem: str):
    tomo_nbr = stem.split('_')[1]
    return str(Path(target_dir) / f"{tomo_nbr}_tomo.nrrd")

for path_ in all_tomos:
    print("processing: ", path_)

    if source_file_format == 'nrrd':
        tomo, _ = nrrd.read(path_)
    elif source_file_format == 'mrc':
        tomo = load_mrc(path_)

    print("Prepping...")
    tomo_dsts = prepare_input(tomo).astype(np.float32)
    tomo_dsts = tomo_dsts.clip(0)
    print("Generating skeleton...")
    tomo_skel = surface_skel(tomo_dsts, f=0)
    tomo_skel = tomo_skel * tomo # get rid of noises
    target_path = get_save_path(target_dir, path_.stem)
    print("Savingskeleton to :", target_path)
    nrrd.write(target_path, tomo_skel.astype(np.uint8))

