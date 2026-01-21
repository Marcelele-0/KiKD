import pytest
import os
import sys
import subprocess
from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent
OUTPUT_DIR = ROOT_DIR / "test_outputs"

@pytest.fixture(autouse=True)
def setup_dirs():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

def run_module(name, args):
    cmd = [sys.executable, "-m", f"src.{name}"] + [str(a) for a in args]
    return subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT_DIR)

def test_hamming_pipeline():
    in_file = OUTPUT_DIR / "input.txt"
    enc_file = OUTPUT_DIR / "encoded.bin"
    noisy_file = OUTPUT_DIR / "noisy.bin"
    dec_file = OUTPUT_DIR / "decoded.txt"
    
    # 1. Create Data
    with open(in_file, "wb") as f:
        f.write(b"Test Message 123")
        
    # 2. Encode
    run_module("encoder", [in_file, enc_file])
    assert enc_file.exists()
    
    # 3. Noise (Small p=0.01 -> Fixable errors)
    run_module("noise", ["0.01", enc_file, noisy_file])
    
    # 4. Decode
    res_dec = run_module("decoder", [noisy_file, dec_file])
    print(res_dec.stdout)
    
    # 5. Check
    res_check = run_module("check", [in_file, dec_file])
    print(res_check.stdout)
    
    # Verify content (should match perfectly with small noise)
    with open(dec_file, "rb") as f:
        assert f.read() == b"Test Message 123"