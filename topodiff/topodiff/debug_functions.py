"""
디버깅 및 데이터 분석 유틸리티 함수들
프로젝트 루트에 debug_functions.py로 저장
"""

import os
import numpy as np
import torch as th
import random

def analyze_dataset_ranges(data_dir, sample_size=100):
    """
    🔧 실제 데이터에서 VF/YM 범위 계산
    """
    print("=== Dataset Range Analysis ===")
    
    # 파일 목록 수집
    implicit_files = [f for f in os.listdir(data_dir) if f.endswith('_implicit.npz')]
    meta_files = [f for f in os.listdir(data_dir) if f.endswith('_meta.npy')]
    
    print(f"Found {len(implicit_files)} implicit files")
    print(f"Found {len(meta_files)} meta files")
    
    # 샘플링하여 범위 계산
    sample_files = random.sample(meta_files, min(sample_size, len(meta_files)))
    
    vf_values = []
    ym_values = []
    
    for meta_file in sample_files:
        meta_path = os.path.join(data_dir, meta_file)
        try:
            meta = np.load(meta_path, allow_pickle=True)
            
            # 배열 형태 처리
            if isinstance(meta, np.ndarray) and meta.shape == (2,):
                vf, ym = meta[0], meta[1]
            elif isinstance(meta, np.ndarray) and meta.dtype == object:
                meta_dict = meta.item()
                vf = meta_dict.get('vf', None)
                ym = meta_dict.get('ym', None)
            else:
                print(f"Unexpected meta format in {meta_file}: {meta}")
                continue
            
            if vf is not None and ym is not None:
                vf_values.append(float(vf))
                ym_values.append(float(ym))
                
        except Exception as e:
            print(f"Error reading {meta_file}: {e}")
            continue
    
    if vf_values and ym_values:
        vf_range = (min(vf_values), max(vf_values))
        ym_range = (min(ym_values), max(ym_values))
        
        print(f"✅ VF range: {vf_range} (mean: {np.mean(vf_values):.3f})")
        print(f"✅ YM range: {ym_range} (mean: {np.mean(ym_values):.1f})")
        print(f"✅ Sample count: {len(vf_values)}")
        
        return vf_range, ym_range
    else:
        print("❌ No valid VF/YM data found")
        return None, None


def test_model_forward(model_path, device='cpu'):
    """
    🔧 모델 forward 테스트
    """
    print("=== Model Forward Test ===")
    
    try:
        from topodiff.script_util import create_model_and_diffusion, model_and_diffusion_defaults
        
        # 모델 생성
        model, diffusion = create_model_and_diffusion(**model_and_diffusion_defaults())
        
        # 가중치 로드
        state_dict = th.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        
        print(f"✅ Model loaded successfully")
        print(f"✅ Parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        # 테스트 입력
        test_input = th.randn(1, 3, 64, 64, 64).to(device)
        test_t = th.tensor([500]).to(device)
        
        print(f"✅ Test input shape: {test_input.shape}")
        
        with th.no_grad():
            output = model(test_input, test_t)
            
        print(f"✅ Model forward successful")
        print(f"✅ Output shape: {output.shape}")
        print(f"✅ Output range: [{output.min():.3f}, {output.max():.3f}]")
        print(f"✅ Contains NaN: {th.isnan(output).any()}")
        
        return True
        
    except Exception as e:
        print(f"❌ Model test failed: {e}")
        return False


def test_data_loading(data_dir):
    """
    🔧 데이터 로딩 테스트
    """
    print("=== Data Loading Test ===")
    
    try:
        from topodiff.image_datasets_diffusion_model import load_data
        
        data = load_data(
            data_dir=data_dir,
            batch_size=2,
            image_size=64,
            split='train',
            num_workers=1
        )
        
        # 첫 배치 가져오기
        batch, cond_dict = next(data)
        
        print(f"✅ Data loading successful")
        print(f"✅ Batch shape: {batch.shape}")
        print(f"✅ Condition keys: {cond_dict.keys()}")
        
        if 'target_vf' in cond_dict:
            vf_vals = cond_dict['target_vf']
            print(f"✅ VF values: {vf_vals}")
        
        if 'target_ym' in cond_dict:
            ym_vals = cond_dict['target_ym']
            print(f"✅ YM values: {ym_vals}")
        
        # 채널별 통계
        print(f"✅ Channel 0 (struct): range=[{batch[:,0].min():.3f}, {batch[:,0].max():.3f}]")
        print(f"✅ Channel 1 (vf): mean={batch[:,1].mean():.3f}, std={batch[:,1].std():.6f}")
        print(f"✅ Channel 2 (ym): mean={batch[:,2].mean():.3f}, std={batch[:,2].std():.6f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Data loading failed: {e}")
        return False


def create_proper_meta_file(data_dir, output_path):
    """
    🔧 적절한 meta.pt 파일 생성
    """
    print("=== Creating Meta File ===")
    
    vf_range, ym_range = analyze_dataset_ranges(data_dir)
    
    if vf_range and ym_range:
        meta = {
            'vf_range': vf_range,
            'ym_range': ym_range
        }
        
        th.save(meta, output_path)
        print(f"✅ Meta file saved to: {output_path}")
        print(f"✅ Content: {meta}")
        return True
    else:
        print("❌ Failed to create meta file")
        return False


def full_debug_pipeline():
    """
    🔧 전체 디버깅 파이프라인
    """
    print("🚀 Starting Full Debug Pipeline")
    
    # 경로 설정
    data_dir = "/home/yeoneung/Euihyun/3D_TPMS_topoDIff/data"
    model_path = "./checkpoints/3d_diff_logdir6/model011000.pt"
    meta_path = "./checkpoints/3d_diff_logdir6/meta.pt"
    
    # 1. 데이터 범위 분석
    vf_range, ym_range = analyze_dataset_ranges(data_dir)
    
    # 2. meta.pt 생성
    if vf_range and ym_range:
        create_proper_meta_file(data_dir, meta_path)
    
    # 3. 모델 테스트
    test_model_forward(model_path)
    
    # 4. 데이터 로딩 테스트
    test_data_loading(data_dir)
    
    print("🎉 Debug pipeline completed!")


if __name__ == "__main__":
    full_debug_pipeline()