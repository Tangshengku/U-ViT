from tools.fid_score import calculate_fid_given_paths
from datasets import get_dataset

dataset = get_dataset(name='celeba',
        path='assets/datasets/celeba',
        resolution=64,)
fid = calculate_fid_given_paths((dataset.fid_stat, "/data/tsk/diff/ddim/image_samples/images"))
print(fid)
