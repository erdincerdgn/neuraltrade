#!/usr/bin/env python3
"""
NeuralTrade Proto File Generator
================================
Regenerates gRPC proto files from ai_service.proto

Usage:
    python scripts/generate_proto.py
    
Requirements:
    pip install grpcio-tools>=1.62.0
"""

import os
import sys
import subprocess
from pathlib import Path


def main():
    # Get project root
    script_dir = Path(__file__).parent.absolute()
    project_root = script_dir.parent
    
    proto_dir = project_root / "proto"
    generated_dir = proto_dir / "generated"
    proto_file = proto_dir / "ai_service.proto"
    
    # Validate paths
    if not proto_file.exists():
        print(f"❌ Proto file not found: {proto_file}")
        sys.exit(1)
    
    # Create generated directory if it doesn't exist
    generated_dir.mkdir(parents=True, exist_ok=True)
    
    # Create __init__.py if it doesn't exist
    init_file = generated_dir / "__init__.py"
    if not init_file.exists():
        init_file.write_text('"""Generated gRPC proto files."""\n')
    
    print(f"📁 Proto dir: {proto_dir}")
    print(f"📁 Generated dir: {generated_dir}")
    print(f"📄 Proto file: {proto_file}")
    print()
    
    # Build command
    cmd = [
        sys.executable, "-m", "grpc_tools.protoc",
        f"-I{proto_dir}",
        f"--python_out={generated_dir}",
        f"--grpc_python_out={generated_dir}",
        str(proto_file)
    ]
    
    print(f"🔧 Running: {' '.join(cmd)}")
    print()
    
    # Execute
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print("✅ Proto files generated successfully!")
        print()
        
        # List generated files
        print("📦 Generated files:")
        for f in generated_dir.glob("*.py"):
            size = f.stat().st_size / 1024
            print(f"   - {f.name} ({size:.1f} KB)")
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Proto generation failed!")
        print(f"   Return code: {e.returncode}")
        if e.stderr:
            print(f"   Error: {e.stderr}")
        sys.exit(1)
    except FileNotFoundError:
        print("❌ grpcio-tools not installed!")
        print("   Run: pip install grpcio-tools>=1.62.0")
        sys.exit(1)


if __name__ == "__main__":
    main()
