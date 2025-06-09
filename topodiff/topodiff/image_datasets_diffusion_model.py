import os
import numpy as np
import torch as th
from torch.utils.data import DataLoader, Dataset


def load_data(*, data_dir, batch_size, image_size, deterministic=False, num_workers=1, train_split=0.8):
    cond_files = sorted([
        os.path.join(data_dir, f) for f in os.listdir(data_dir) 
        if f.endswith('_cond.npz')
    ])
    
    # Train/Test 분할
    n_train = int(len(cond_files) * train_split)
    if deterministic:
        # Test mode: 뒤의 20% 사용
        cond_files = cond_files[n_train:]
    else:
        # Train mode: 앞의 80% 사용
        cond_files = cond_files[:n_train]
    
    if not cond_files:
        raise ValueError(f"No files found for {'training' if not deterministic else 'testing'}")
    
    # Create dataset
    dataset = VolumeDataset(image_size, cond_files)
    
    # Create dataloader
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=not deterministic,
        num_workers=num_workers,
        drop_last=True,
        pin_memory=th.cuda.is_available()
    )
    
    # Infinite generator
    while True:
        yield from loader


class VolumeDataset(Dataset):
    """
    Dataset for 3D volumes with VF and YM conditions.
    Each sample contains:
    - volume[0]: Structure field (normalized to [-1, 1])
    - volume[1]: Volume fraction (constant across volume)
    - volume[2]: Young's modulus (constant across volume)
    """
    
    def __init__(self, resolution, volume_paths, vf_range=(0.1, 0.9), ym_range=(1.0, 100.0)):
        super().__init__()
        self.resolution = resolution
        self.volume_paths = volume_paths
        self.vf_range = vf_range
        self.ym_range = ym_range

    def __len__(self):
        return len(self.volume_paths)

    def __getitem__(self, idx):
        # Load volume data
        volume_path = self.volume_paths[idx]
        with np.load(volume_path) as data:
            # Expecting shape (D, H, W, 3) where:
            # channel 0: structure field
            # channel 1: volume fraction (scalar expanded to volume)
            # channel 2: young's modulus (scalar expanded to volume)
            arr = data['arr_0'].astype(np.float32)
            
            # Extract metadata if available
            vf = data.get('vf', None)
            ym = data.get('ym', None)
        
        # Reshape from (D, H, W, C) to (C, D, H, W)
        volume = np.transpose(arr, (3, 0, 1, 2))
        
        # Normalize structure channel to [-1, 1]
        structure = volume[0]
        vmin, vmax = structure.min(), structure.max()
        if vmax > vmin:
            structure = 2 * (structure - vmin) / (vmax - vmin) - 1
        else:
            structure = np.zeros_like(structure)
        volume[0] = structure
        
        # If VF/YM not in metadata, extract from volume
        if vf is None:
            vf = volume[1, 0, 0, 0]  # Assuming constant across volume
        if ym is None:
            ym = volume[2, 0, 0, 0]  # Assuming constant across volume
            
        # Normalize VF and YM to [0, 1] range for network stability
        vf_norm = (vf - self.vf_range[0]) / (self.vf_range[1] - self.vf_range[0])
        ym_norm = (ym - self.ym_range[0]) / (self.ym_range[1] - self.ym_range[0])
        
        # Update condition channels with normalized values
        volume[1] = np.full_like(volume[1], vf_norm)
        volume[2] = np.full_like(volume[2], ym_norm)
        
        # Return volume and metadata
        return volume, {'vf': vf, 'ym': ym, 'vf_norm': vf_norm, 'ym_norm': ym_norm}


def create_synthetic_data(num_samples, resolution, save_dir):
    """
    Create synthetic data for testing.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    for i in range(num_samples):
        # Random VF and YM
        vf = np.random.uniform(0.1, 0.9)
        ym = np.random.uniform(1.0, 100.0)
        
        # Create synthetic structure (e.g., TPMS-like)
        x = np.linspace(-np.pi, np.pi, resolution)
        y = np.linspace(-np.pi, np.pi, resolution)
        z = np.linspace(-np.pi, np.pi, resolution)
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        
        # Gyroid-like structure
        structure = np.sin(X) * np.cos(Y) + np.sin(Y) * np.cos(Z) + np.sin(Z) * np.cos(X)
        structure = structure + vf * 2 - 1  # Adjust based on VF
        
        # Create volume with 3 channels
        volume = np.zeros((resolution, resolution, resolution, 3), dtype=np.float32)
        volume[..., 0] = structure
        volume[..., 1] = vf
        volume[..., 2] = ym
        
        # Save
        filename = os.path.join(save_dir, f'volume_{i:06d}_cond.npz')
        np.savez_compressed(filename, arr_0=volume, vf=vf, ym=ym)
        
    print(f"Created {num_samples} synthetic samples in {save_dir}")