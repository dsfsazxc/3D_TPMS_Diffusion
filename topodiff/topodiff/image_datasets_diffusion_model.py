import math
import random

from PIL import Image
import blobfile as bf
from mpi4py import MPI
import numpy as np
from torch.utils.data import DataLoader, Dataset


def load_data(
    *,
    data_dir,
    batch_size,
    image_size,
    deterministic=False,
):
    """
    3D 볼륨 데이터 로더
    """
    if not data_dir:
        raise ValueError("unspecified data directory")
    
    all_volumes = _list_volume_files_recursively(data_dir)
    
    dataset = VolumeDataset(
        image_size,  # 3D에서도 같은 크기 사용
        all_volumes,
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

def _list_volume_files_recursively(data_dir):
    volumes = []
    for entry in sorted(bf.listdir(data_dir)):
        full_path = bf.join(data_dir, entry)
        ext = entry.split(".")[-1]
        if "." in entry and ext.lower() in ["npz"]:
            volumes.append(full_path)
        elif bf.isdir(full_path):
            volumes.extend(_list_volume_files_recursively(full_path))
    return volumes

class VolumeDataset(Dataset):
    def __init__(self, resolution, volume_paths, shard=0, num_shards=1):
        super().__init__()
        self.resolution = resolution
        self.local_volumes = volume_paths[shard:][::num_shards]
    
    def __len__(self):
        return len(self.local_volumes)
    
    def __getitem__(self, idx):
        volume_path = self.local_volumes[idx]
        
        # npz 파일에서 arr_0 로드
        with np.load(volume_path) as data:
            volume = data['arr_0'].astype(np.float32)
        
        # 값을 -1~1로 정규화
        volume_min = volume.min()
        volume_max = volume.max()
        if volume_max > volume_min:
            volume = 2 * (volume - volume_min) / (volume_max - volume_min) - 1
        else:
            volume = volume * 0
        
        # [D, H, W] -> [1, D, H, W]
        volume = volume[np.newaxis, ...]
        
        # 빈 제약조건 생성 (0개 채널)
        dummy_cons = np.zeros((0, self.resolution, self.resolution, self.resolution), dtype=np.float32)
        
        out_dict = {}
        return volume, dummy_cons, out_dict  # 3개 반환
    
    def resize_volume(self, volume, target_size):
        # 간단한 3D 리사이징 (실제로는 더 정교한 방법 필요)
        from scipy.ndimage import zoom
        zoom_factors = [target_size/s for s in volume.shape]
        return zoom(volume, zoom_factors, order=1)
        
class ImageDataset(Dataset):
    def __init__(
        self,
        resolution,
        image_paths,
        constraint_pf_paths,
        loads_paths,
        shard=0,
        num_shards=1
    ):
        super().__init__()
        self.resolution = resolution
        self.local_images = image_paths[shard:][::num_shards]
        self.local_constraints_pf = constraint_pf_paths[shard:][::num_shards]
        self.local_loads = loads_paths[shard:][::num_shards]

    def __len__(self):
        return len(self.local_images)

    def __getitem__(self, idx):
        volume_path = self.local_volumes[idx]  # 3D 볼륨 경로

        num_im = int((image_path.split("_")[-1]).split(".")[0])
        num_cons_pf = int((constraint_pf_path.split("_")[-1]).split(".")[0])
        num_load = int((load_path.split("_")[-1]).split(".")[0])
        assert num_im == num_cons_pf == num_load, "Problem while loading the images and constraints"

        # 3D 볼륨 데이터 로드 (이미 -1~1 범위라고 가정)
        volume = np.load(volume_path).astype(np.float32)
        
        # 크기 조정이 필요한 경우
        if volume.shape != (self.resolution, self.resolution, self.resolution):
            volume = self.resize_volume(volume, self.resolution)
        
        # [D, H, W, C] -> [C, D, H, W] 형태로 변환
        volume = volume.reshape(self.resolution, self.resolution, self.resolution, 1)
        volume = np.transpose(volume, [3, 0, 1, 2])

        constraints_pf = np.load(constraint_pf_path)
        assert constraints_pf.shape[0:2] == arr.shape[0:2], "The constraints do not fit the dimension of the image"

        loads = np.load(load_path)
        assert loads.shape[0:2] == arr.shape[0:2], "The constraints do not fit the dimension of the image"

        constraints = np.concatenate([constraints_pf, loads], axis = 2)

        out_dict = {}
        return np.transpose(arr, [2, 0, 1]).astype(np.float32), np.transpose(constraints, [2, 0, 1]).astype(np.float32), out_dict

def center_crop_arr(pil_image, image_size):
    # We are not on a new enough PIL to support the `reducing_gap`
    # argument, which uses BOX downsampling at powers of two first.
    # Thus, we do it by hand to improve downsample quality.
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size]