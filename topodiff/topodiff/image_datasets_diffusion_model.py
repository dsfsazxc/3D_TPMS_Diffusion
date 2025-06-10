import os
import numpy as np
import torch as th
from torch.utils.data import DataLoader, Dataset


def load_data(*, data_dir, batch_size, image_size, deterministic=False, num_workers=1, train_split=0.8):
    """Load 3D structure data for diffusion training"""
    
    # Find all .npz files in data directory
    data_files = []
    for f in os.listdir(data_dir):
        if f.endswith('.npz'):
            data_files.append(os.path.join(data_dir, f))
    
    data_files = sorted(data_files)
    
    if not data_files:
        raise ValueError(f"No .npz files found in {data_dir}")
    
    # Train/Test split
    n_train = int(len(data_files) * train_split)
    if deterministic:
        # Test mode: use last 20%
        data_files = data_files[n_train:]
    else:
        # Train mode: use first 80%
        data_files = data_files[:n_train]
    
    if not data_files:
        raise ValueError(f"No files found for {'training' if not deterministic else 'testing'}")
    
    print(f"Loading {len(data_files)} files for {'training' if not deterministic else 'testing'}")
    
    # Create dataset
    dataset = VolumeDataset(image_size, data_files)
    
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
    Dataset for 3D structures.
    Returns normalized 3D volumes in range [-1, 1].
    """
    
    def __init__(self, resolution, volume_paths):
        super().__init__()
        self.resolution = resolution
        self.volume_paths = volume_paths
        print(f"VolumeDataset initialized with {len(volume_paths)} volumes, target resolution: {resolution}")

    def __len__(self):
        return len(self.volume_paths)

    def __getitem__(self, idx):
        volume_path = self.volume_paths[idx]
        
        try:
            # Load volume data
            with np.load(volume_path) as data:
                # Handle different possible key names
                if 'arr_0' in data:
                    volume = data['arr_0'].astype(np.float32)
                elif 'volume' in data:
                    volume = data['volume'].astype(np.float32)
                else:
                    # Use first available array
                    key = list(data.keys())[0]
                    volume = data[key].astype(np.float32)
            
            # Handle different input shapes
            if volume.ndim == 3:
                # Shape: (D, H, W) -> add channel dimension
                volume = volume[np.newaxis, ...]  # (1, D, H, W)
            elif volume.ndim == 4:
                if volume.shape[-1] in [1, 3]:
                    # Shape: (D, H, W, C) -> (C, D, H, W)
                    volume = np.transpose(volume, (3, 0, 1, 2))
                    # Take only first channel if multiple channels
                    volume = volume[0:1]
                elif volume.shape[0] in [1, 3]:
                    # Already in (C, D, H, W) format
                    volume = volume[0:1]  # Take only first channel
                else:
                    raise ValueError(f"Unexpected 4D volume shape: {volume.shape}")
            else:
                raise ValueError(f"Unexpected volume shape: {volume.shape}")
            
            # Ensure we have single channel
            if volume.shape[0] != 1:
                volume = volume[0:1]
            
            # Resize if needed
            if volume.shape[1:] != (self.resolution, self.resolution, self.resolution):
                # Simple resize using numpy (for more sophisticated resizing, use scipy)
                volume = self._resize_volume(volume[0], self.resolution)
                volume = volume[np.newaxis, ...]
            
            # Normalize to [-1, 1]
            vmin, vmax = volume.min(), volume.max()
            if vmax > vmin:
                volume = 2 * (volume - vmin) / (vmax - vmin) - 1
            else:
                volume = np.zeros_like(volume)
            
            return th.from_numpy(volume)
            
        except Exception as e:
            print(f"Error loading {volume_path}: {e}")
            # Return dummy data if loading fails
            return th.randn(1, self.resolution, self.resolution, self.resolution)
    
    def _resize_volume(self, volume, target_size):
        """Simple volume resizing using numpy interpolation"""
        from scipy.ndimage import zoom
        
        current_shape = volume.shape
        zoom_factors = [target_size / s for s in current_shape]
        
        return zoom(volume, zoom_factors, order=1)


def create_synthetic_data(num_samples, resolution, save_dir):
    """
    Create synthetic 3D TPMS data for testing.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"Creating {num_samples} synthetic samples in {save_dir}")
    
    for i in range(num_samples):
        # Create synthetic TPMS-like structure
        x = np.linspace(-np.pi, np.pi, resolution)
        y = np.linspace(-np.pi, np.pi, resolution)
        z = np.linspace(-np.pi, np.pi, resolution)
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        
        # Gyroid structure with some randomness
        phase = np.random.uniform(0, 2*np.pi)
        scale = np.random.uniform(0.8, 1.2)
        
        structure = (np.sin(X + phase) * np.cos(Y) + 
                    np.sin(Y + phase) * np.cos(Z) + 
                    np.sin(Z + phase) * np.cos(X)) * scale
        
        # Add some noise
        noise = np.random.normal(0, 0.1, structure.shape)
        structure = structure + noise
        
        # Save as single channel volume
        filename = os.path.join(save_dir, f'volume_{i:06d}.npz')
        np.savez_compressed(filename, arr_0=structure.astype(np.float32))
        
        if i % 100 == 0:
            print(f"Created {i+1}/{num_samples} samples")
    
    print(f"Created {num_samples} synthetic samples in {save_dir}")


if __name__ == "__main__":
    # Test the dataset
    create_synthetic_data(100, 64, "./test_data")
    
    # Test loading
    data_loader = load_data(
        data_dir="./test_data",
        batch_size=2,
        image_size=64,
        deterministic=False,
        num_workers=1
    )
    
    batch = next(data_loader)
    print(f"Loaded batch shape: {batch.shape}")
    print(f"Value range: [{batch.min():.3f}, {batch.max():.3f}]")