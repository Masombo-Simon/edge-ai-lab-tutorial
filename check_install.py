# check_install.py
# Run this FIRST before anything else

print("=" * 55)
print("  CHECKING YOUR INSTALLATION")
print("=" * 55)

# ── Test 1: PyTorch ────────────────────────────────────
try:
    import torch
    print(f"\n✅ PyTorch:       {torch.__version__}")
except ImportError:
    print("\n❌ PyTorch:       NOT INSTALLED")
    print("   Fix: pip install torch")

# ── Test 2: Torchvision ────────────────────────────────
try:
    import torchvision
    print(f"✅ Torchvision:   {torchvision.__version__}")
except ImportError:
    print("❌ Torchvision:   NOT INSTALLED")
    print("   Fix: pip install torchvision")

# ── Test 3: TI edgeai-torchvision ─────────────────────
try:
    import edgeai_torchvision
    print(f"✅ TI edgeai-torchvision: INSTALLED ← Real TI library!")
    TI_AVAILABLE = True
except ImportError:
    print("⚠️  TI edgeai-torchvision: NOT INSTALLED")
    print("   Using standard torchvision instead")
    print("   (Concepts are identical)")
    TI_AVAILABLE = False

# ── Test 4: Other packages ─────────────────────────────
packages = {
    "numpy":      "numpy",
    "pandas":     "pandas",
    "matplotlib": "matplotlib",
    "cv2":        "opencv-python",
    "PIL":        "pillow",
    "onnx":       "onnx",
    "onnxruntime":"onnxruntime",
}

print()
for import_name, pip_name in packages.items():
    try:
        __import__(import_name)
        import importlib
        mod = importlib.import_module(import_name)
        ver = getattr(mod, '__version__', 'installed')
        print(f"✅ {import_name:<15} {ver}")
    except ImportError:
        print(f"❌ {import_name:<15} NOT INSTALLED")
        print(f"   Fix: pip install {pip_name}")

# ── Test 5: Quick model load ───────────────────────────
print("\n" + "-" * 55)
print("Testing model load...")

try:
    if TI_AVAILABLE:
        from edgeai_torchvision import models
        model = models.mobilenet_v2(pretrained=False)
        print("✅ TI Model loaded successfully!")
    else:
        from torchvision import models
        model = models.mobilenet_v2(weights=None)
        print("✅ Standard model loaded successfully!")
except Exception as e:
    print(f"❌ Model load failed: {e}")

# ── Summary ────────────────────────────────────────────
print("\n" + "=" * 55)
print(f"  TI edgeai-torchvision available: {TI_AVAILABLE}")
if TI_AVAILABLE:
    print("  ✅ Run demos with REAL TI models!")
else:
    print("  ✅ Run demos with compatibility wrapper")
    print("  (Results and concepts are identical)")
print("=" * 55)