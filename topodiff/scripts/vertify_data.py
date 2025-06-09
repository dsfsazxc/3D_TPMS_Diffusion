# scripts/verify_data.py
import numpy as np
from torch.utils.data import DataLoader, Dataset

class VolumeDataset(Dataset):
    def __init__(self, resolution, volume_paths, shard=0, num_shards=1):
        super().__init__()
        self.resolution = resolution
        self.local_volumes = volume_paths[shard::num_shards]

    def __len__(self):
        return len(self.local_volumes)

    def __getitem__(self, idx):
        volume_path = self.local_volumes[idx]
        with np.load(volume_path) as data:
            arr = data['arr_0'].astype(np.float32)  # (D,H,W,3)
        # 채널 마지막 → 채널 첫 번째
        volume = np.transpose(arr, (3,0,1,2))     # (3,D,H,W)
        # [-1,1] 정규화 (첫 채널만)
        v = volume[0]
        vmin, vmax = v.min(), v.max()
        if vmax>vmin:
            v = 2*(v - vmin)/(vmax-vmin) - 1
        else:
            v = v*0
        volume[0] = v
        # VF/YM 채널(1,2)는 이미 상수값 → 유지
        # dummy out_dict
        return volume, np.zeros((0,*volume.shape[1:]), dtype=np.float32), {}

def load_data(*, data_dir, batch_size, image_size, deterministic=False):
    import os
    from torch.utils.data import DataLoader
    # 1) 조건부 npz 파일 경로 수집
    cond_files = sorted(
        [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('_cond.npz')]
    )
    # 2) 데이터로더 생성
    dataset = VolumeDataset(image_size, cond_files)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=not deterministic,
        num_workers=4,
        drop_last=True
    )
    # 3) 무한 반복 제너레이터
    while True:
        yield from loader

data_dir = "/home/yeoneung/Euihyun/3D_TPMS_topoDIff/data"
batch_size = 2
image_size = 64

data_iter = load_data(data_dir=data_dir, batch_size=batch_size, image_size=image_size, deterministic=True)
volumes, _, _ = next(data_iter)

for i in range(volumes.shape[1]):
    ch = volumes[0, i]
    print(f"Channel {i}: shape={ch.shape}, min={ch.min():.4f}, max={ch.max():.4f}, mean={ch.mean():.4f}")
    ch_np = ch.numpy()
    if np.allclose(ch_np, ch_np.flatten()[0]):
        print(f"  -> Constant channel with value {ch_np.flatten()[0]:.4f}")
