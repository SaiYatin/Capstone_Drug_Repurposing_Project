"""
GPU Configuration Test
======================

Verifies that PyTorch can see and use the RTX 4090 GPU.

Run this BEFORE starting training to ensure everything is configured correctly.
"""

import torch
import sys

print("="*80)
print("GPU CONFIGURATION TEST")
print("="*80)

# 1. Check Python version
print(f"\n1. Python Version: {sys.version}")

# 2. Check PyTorch version
print(f"\n2. PyTorch Version: {torch.__version__}")

# 3. Check CUDA availability
cuda_available = torch.cuda.is_available()
print(f"\n3. CUDA Available: {cuda_available}")

if not cuda_available:
    print("\n❌ ERROR: CUDA not available!")
    print("\nPossible fixes:")
    print("  1. Install NVIDIA drivers")
    print("  2. Install CUDA Toolkit")
    print("  3. Reinstall PyTorch with CUDA:")
    print("     pip install torch --index-url https://download.pytorch.org/whl/cu118")
    sys.exit(1)

# 4. GPU Information
print(f"\n4. GPU Device:")
print(f"   Name: {torch.cuda.get_device_name(0)}")
print(f"   Count: {torch.cuda.device_count()}")
print(f"   Current Device: {torch.cuda.current_device()}")

# 5. CUDA Version
print(f"\n5. CUDA Version: {torch.version.cuda}")

# 6. Memory Information
gpu_props = torch.cuda.get_device_properties(0)
total_memory_gb = gpu_props.total_memory / (1024**3)
print(f"\n6. GPU Memory:")
print(f"   Total: {total_memory_gb:.1f} GB")
print(f"   Allocated: {torch.cuda.memory_allocated(0) / (1024**3):.1f} GB")
print(f"   Cached: {torch.cuda.memory_reserved(0) / (1024**3):.1f} GB")

# 7. Test GPU computation
print(f"\n7. Testing GPU Computation...")
try:
    x = torch.randn(1000, 1000).cuda()
    y = torch.randn(1000, 1000).cuda()
    z = torch.matmul(x, y)
    print("   ✓ GPU computation successful!")
except Exception as e:
    print(f"   ✗ GPU computation failed: {e}")
    sys.exit(1)

# 8. Mixed Precision Support
print(f"\n8. Mixed Precision (FP16) Support:")
if torch.cuda.get_device_capability(0)[0] >= 7:
    print("   ✓ Supported (Tensor Cores available)")
else:
    print("   ⚠ Not optimal (but will work)")

# 9. Recommended Configuration
print(f"\n9. Recommended Configuration for {torch.cuda.get_device_name(0)}:")
if "4090" in torch.cuda.get_device_name(0):
    print("   Hidden Dim: 512")
    print("   Batch Size: 64")
    print("   Num Epochs: 100")
    print("   Mixed Precision: True")
    print("   Expected AUPRC: 0.88-0.92")
elif "V100" in torch.cuda.get_device_name(0):
    print("   Hidden Dim: 512")
    print("   Batch Size: 48")
    print("   Num Epochs: 100")
elif "A100" in torch.cuda.get_device_name(0):
    print("   Hidden Dim: 768")
    print("   Batch Size: 128")
    print("   Num Epochs: 150")
else:
    print("   Hidden Dim: 256")
    print("   Batch Size: 32")
    print("   Num Epochs: 50")

print("\n" + "="*80)
print("✓✓✓ ALL CHECKS PASSED - GPU IS READY! ✓✓✓")
print("="*80)
print("\nYou can now run:")
print("  cd training")
print("  python train_optimized.py")
