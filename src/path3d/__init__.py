import os

# unbuffered stdout/stderr for real-time log streaming, and guard against the duplicate-OpenMP-runtime abort
# thread counts are intentionally not pinned here pinned here -- the HPC scheduler/environment decides

os.environ.setdefault("PYTHONUNBUFFERED", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")