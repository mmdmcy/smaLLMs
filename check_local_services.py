#!/usr/bin/env python3
"""
Quick diagnostic tool to check local AI services
"""
import requests
import subprocess
import json
from pathlib import Path

def check_ollama():
    """Check if Ollama is running and list models."""
    print(" Checking Ollama...")
    try:
        # Try command line first
        result = subprocess.run(['ollama', 'list'], capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')[1:]  # Skip header
            models = [line.split()[0] for line in lines if line.strip()]
            print(f" Ollama: Found {len(models)} models via command line")
            for model in models[:5]:  # Show first 5
                print(f"    {model}")
            if len(models) > 5:
                print(f"   ... and {len(models) - 5} more")
            return True, models
        else:
            print(f" Ollama command failed: {result.stderr}")
            return False, []
    except subprocess.TimeoutExpired:
        print(" Ollama command timed out")
        return False, []
    except FileNotFoundError:
        print(" Ollama not found in PATH")
        return False, []
    except Exception as e:
        print(f" Ollama error: {e}")
        return False, []

def check_lm_studio():
    """Check if LM Studio server is running."""
    print("\n Checking LM Studio...")
    ports_to_try = [1234, 1235, 8080, 8000]
    
    for port in ports_to_try:
        try:
            url = f"http://localhost:{port}/v1/models"
            print(f"   Trying port {port}...")
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                data = response.json()
                models = data.get('data', [])
                print(f" LM Studio: Found server on port {port} with {len(models)} models")
                for model in models:
                    name = model.get('id', 'unknown')
                    print(f"    {name}")
                return True, port, models
            else:
                print(f"   Port {port}: HTTP {response.status_code}")
        except requests.exceptions.ConnectTimeoutError:
            print(f"   Port {port}: Connection timeout")
        except requests.exceptions.ConnectionError:
            print(f"   Port {port}: Connection refused")
        except Exception as e:
            print(f"   Port {port}: {e}")
    
    print(" LM Studio: No server found on common ports")
    print("   Make sure LM Studio is running with 'Start Server' enabled")
    return False, None, []

def check_config():
    """Check current configuration."""
    print("\n Checking Configuration...")
    config_path = Path("config/config.yaml")
    if config_path.exists():
        print(" Config file found")
        # Could parse YAML here if needed
    else:
        print("  No config file found, using defaults")

def main():
    print(" smaLLMs Local Services Diagnostic\n")
    
    # Check services
    ollama_ok, ollama_models = check_ollama()
    lms_ok, lms_port, lms_models = check_lm_studio()
    check_config()
    
    # Summary
    print(f"\n Summary:")
    print(f"   Ollama: {' Working' if ollama_ok else ' Not working'} ({len(ollama_models)} models)")
    print(f"   LM Studio: {' Working' if lms_ok else ' Not working'} ({len(lms_models) if lms_ok else 0} models)")
    
    if not lms_ok:
        print(f"\n To fix LM Studio:")
        print("   1. Open LM Studio application")
        print("   2. Go to 'Local Server' or 'Server' tab")
        print("   3. Click 'Start Server'")
        print("   4. Make sure a model is loaded")
        print("   5. Server should show 'Running on http://localhost:1234'")
    
    if ollama_ok or lms_ok:
        print(f"\n Ready to run: python smaLLMs.py")
        if ollama_ok and not lms_ok:
            print("   Use 'ollama' command for Ollama-only evaluation")
        elif lms_ok and not ollama_ok:
            print("   Use 'lms' command for LM Studio-only evaluation")
        else:
            print("   Use 'local' command for all local models")

if __name__ == "__main__":
    main()
