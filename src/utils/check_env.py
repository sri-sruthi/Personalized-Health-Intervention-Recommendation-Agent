# check_env.py
import sys
import os
import importlib.util
import site

print("=" * 70)
print("🔍 PYTHON ENVIRONMENT CHECK")
print("=" * 70)
print(f"🧩 Python executable: {sys.executable}")
print(f"📁 Current working directory: {os.getcwd()}")
print(f"🐍 Python version: {sys.version}")
print()

print("📦 Site-packages paths:")
for path in site.getsitepackages():
    print(f"   - {path}")
print()

# Check tensorflow installation
print("🔬 TensorFlow status:")
tf_spec = importlib.util.find_spec("tensorflow")
if tf_spec is None:
    print("❌ TensorFlow NOT found in this environment.")
else:
    import tensorflow as tf
    print(f"✅ TensorFlow imported successfully!")
    print(f"   Version: {tf.__version__}")
    print(f"   Path: {tf.__file__}")

print()
print("=" * 70)
print("✅ Environment check complete.")
print("=" * 70)

