import numpy as np
from mpi4py import MPI
from torch.utils.data import DataLoader, Dataset
import blobfile as bf
import torch as th

def load_data(
    *,
    data_dir,
    batch_size,
    image_size,
    deterministic=False,
):
    """
    3D 볼륨과 대응하는 VF, Young's Modulus 데이터 로더
    """
    if not data_dir:
        raise ValueError("unspecified data directory")
    
    all_volumes, all_properties = _list_volume_property_files(data_dir)
    
    dataset = VolumePropertyDataset(
        image_size,
        all_volumes,
        all_properties,
        shard=MPI.COMM_WORLD.Get_rank(),
        num_shards=MPI.COMM_WORLD.Get_size()
    )
    
    if deterministic:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=1, drop_last=True
        )
    else:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, num_workers=1, drop_last=True
        )
    
    while True:
        yield from loader

def _list_volume_property_files(data_dir):
    """볼륨 파일과 대응하는 속성 파일 찾기"""
    volumes = []
    properties = []
    
    for entry in sorted(bf.listdir(data_dir)):
        if entry.endswith('_implicit.npz'):
            volume_path = bf.join(data_dir, entry)
            property_path = bf.join(data_dir, entry.replace('_implicit.npz', '_meta.npy'))
            
            if bf.exists(property_path):
                volumes.append(volume_path)
                properties.append(property_path)
    
    return volumes, properties

import numpy as np
import torch
from torch.utils.data import Dataset

class VoxelPropertyDataset(Dataset):
    def __init__(self, volume_paths, property_file):
        self.volume_paths = volume_paths
        self.properties = np.load(property_file)

        vf = self.properties[:, 0]
        ym = self.properties[:, 1]

        # 정규화만 하고 저장 안 함
        self.min_vf = vf.min()
        self.max_vf = vf.max()
        self.mean_ym = ym.mean()
        self.std_ym = ym.std()

        self.vf_norm = (vf - self.min_vf) / (self.max_vf - self.min_vf + 1e-8)
        self.ym_norm = (ym - self.mean_ym) / (self.std_ym + 1e-8)
        self.normalized_properties = np.stack([self.vf_norm, self.ym_norm], axis=1)

    def __len__(self):
        return len(self.volume_paths)

    def __getitem__(self, idx):
        # 볼륨 데이터 불러오기
        volume = np.load(self.volume_paths[idx])  # shape: [64, 64, 64]
        volume = torch.tensor(volume, dtype=torch.float32).unsqueeze(0)  # shape: [1, 64, 64, 64]
        target = torch.tensor(self.normalized_properties[idx], dtype=torch.float32)  # shape: [2]
        return volume, target
