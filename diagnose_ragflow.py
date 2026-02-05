#!/usr/bin/env python3
"""
RAGFlow Integration Diagnostics Tool

This script helps you understand why your queries aren't triggering rerank.
"""

import requests
import json
from datetime import datetime

print("=" * 80)
print("RAGFlow + EBM Rerank Integration Diagnostics")
print("=" * 80)
print()

# Test 1: Check if API is running
print("Test 1: API Health Check")
print("-" * 80)
try:
    resp = requests.get("http://127.0.0.1:8000/health", timeout=5)
    if resp.status_code == 200:
        data = resp.json()
        print(f"✅ API is running")
        print(f"   Status: {data['status']}")
        print(f"   Device: {data['device']}")
        print(f"   Model loaded: {data['model_loaded']}")
    else:
        print(f"❌ API returned status {resp.status_code}")
        exit(1)
except requests.exceptions.ConnectionError:
    print("❌ Cannot connect to API at http://127.0.0.1:8000")
    print("   Make sure your FastAPI server is running:")
    print("   uvicorn ebm.main_optimized:app --host 0.0.0.0 --port 8000")
    exit(1)
except Exception as e:
    print(f"❌ Error: {e}")
    exit(1)

print()

# Test 2: Test rerank endpoint directly
print("Test 2: Direct Rerank Test")
print("-" * 80)
test_req = {
    "query": "machine learning",
    "documents": [
        "ML is great",
        "Weather is sunny",
        "Deep learning rocks"
    ],
    "top_n": 3
}

try:
    resp = requests.post("http://127.0.0.1:8000/v1/rerank", json=test_req, timeout=10)
    if resp.status_code == 200:
        data = resp.json()
        print(f"✅ Rerank endpoint works")
        print(f"   Returned {len(data['results'])} results")
        print("   Top result:")
        if data['results']:
            top = data['results'][0]
            print(f"      Index: {top['index']}")
            print(f"      Score: {top['relevance_score']:.4f}")
    else:
        print(f"❌ Rerank failed with status {resp.status_code}")
        print(f"   Response: {resp.text}")
except Exception as e:
    print(f"❌ Error: {e}")

print()

# Test 3: Test endpoint for easy debugging
print("Test 3: Built-in Test Endpoint")
print("-" * 80)
try:
    resp = requests.post("http://127.0.0.1:8000/test-rerank", timeout=10)
    if resp.status_code == 200:
        data = resp.json()
        print(f"✅ Test endpoint works")
        print(f"   Query: {data['test_query']}")
        print("   Top 3 results:")
        for r in data['results'][:3]:
            print(f"      {r['rank']}. [{r['index']}] score={r['score']:.4f}")
            print(f"         {r['document'][:60]}...")
    else:
        print(f"❌ Test failed: {resp.status_code}")
except Exception as e:
    print(f"❌ Error: {e}")

print()

# Test 4: Check RAGFlow configuration
print("Test 4: RAGFlow Configuration Check")
print("-" * 80)
print("Checking RAGFlow configuration files...")

# Check service_conf.yaml
try:
    with open("/media/bazzi/Bazzi/GitHubBazzi/ragflow/conf/service_conf.yaml", "r") as f:
        content = f.read()
        if "EBM" in content and "ebm-rerank" in content:
            print("✅ service_conf.yaml has EBM configuration")
            # Extract rerank model config
            lines = content.split('\n')
            in_rerank = False
            for line in lines:
                if 'rerank_model:' in line:
                    in_rerank = True
                if in_rerank and any(x in line for x in ['factory', 'base_url', 'model_name']):
                    print(f"   {line.strip()}")
                if in_rerank and line.strip() and not line.strip().startswith((' ', '-', 'factory', 'api_key', 'base_url', 'model_name')):
                    break
        else:
            print("⚠️  service_conf.yaml doesn't mention EBM")
except FileNotFoundError:
    print("⚠️  Cannot find service_conf.yaml")
except Exception as e:
    print(f"⚠️  Error reading config: {e}")

print()

# Test 5: Check if RAGFlow backend is running
print("Test 5: RAGFlow Backend Status")
print("-" * 80)
try:
    import subprocess
    result = subprocess.run(
        ["ps", "aux"],
        capture_output=True,
        text=True
    )
    if "ragflow_server" in result.stdout:
        print("✅ RAGFlow backend is running")
        # Extract process info
        for line in result.stdout.split('\n'):
            if "ragflow_server.py" in line:
                parts = line.split()
                pid = parts[1]
                cpu = parts[2]
                mem = parts[3]
                print(f"   PID: {pid}, CPU: {cpu}%, MEM: {mem}%")
    else:
        print("❌ RAGFlow backend not running")
        print("   Start it with: bash docker/launch_backend_service.sh")
except Exception as e:
    print(f"⚠️  Cannot check processes: {e}")

print()

# Summary
print("=" * 80)
print("📊 Diagnosis Summary")
print("=" * 80)
print()
print("If all tests passed but you still don't see rerank calls:")
print()
print("1. ❓ Does your knowledge base have documents?")
print("   → Go to RAGFlow UI → Knowledge Base → Check document count")
print()
print("2. ❓ Are documents processed?")
print("   → Documents must be in 'processed' state, not 'processing'")
print()
print("3. ❓ Does your query match any documents?")
print("   → Try a query that includes words from your documents")
print()
print("4. ❓ Is EBM configured in the UI?")
print("   → Settings → Model Providers → EBM should be configured")
print()
print("5. ❓ Is reranking enabled for your knowledge base?")
print("   → Knowledge Base Settings → Enable reranking → Select EBM")
print()
print("6. ❓ Are you using a NEW chat session?")
print("   → Old chats might not use new configuration")
print()
print("=" * 80)
print("Next Steps:")
print("1. If EBM API tests pass → The problem is RAGFlow configuration")
print("2. Check RAGFlow logs: tail -f /media/bazzi/Bazzi/GitHubBazzi/ragflow/logs/ragflow_server.log")
print("3. Watch API logs while querying RAGFlow")
print("4. Make sure your query retrieves documents (no docs = no rerank)")
print("=" * 80)
