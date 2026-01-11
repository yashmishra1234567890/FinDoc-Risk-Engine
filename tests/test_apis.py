
import requests
import time
import sys

BASE_URL = "http://127.0.0.1:8002"

def check_health():
    try:
        print("Checking /health...")
        r = requests.get(f"{BASE_URL}/health")
        if r.status_code == 200:
            print("✅ /health is UP")
            return True
        else:
            print(f"❌ /health failed: {r.status_code}")
            return False
    except Exception as e:
        print(f"❌ /health connection error: {e}")
        return False

def check_root():
    try:
        print("Checking root / ...")
        r = requests.get(f"{BASE_URL}/")
        if r.status_code == 200:
            print("✅ Root / is UP")
            return True
        else:
            print(f"❌ Root / failed: {r.status_code}")
            return False
    except Exception as e:
        print(f"❌ Root / connection error: {e}")
        return False

def check_query():
    print("Checking /query (Mock)...")
    payload = {"question": "What is the total debt?"}
    try:
        # We expect a mock response or a real one depending on loaded state
        # Since we just started server fresh, FAISS might be empty or loaded from disk.
        # This test ensures the endpoint accepts the request.
        r = requests.post(f"{BASE_URL}/query", json=payload)
        if r.status_code == 200:
            print("✅ /query is UP")
            print(f"   Response: {r.json()}")
            return True
        else:
            print(f"❌ /query failed: {r.status_code} - {r.text}")
            return False
    except Exception as e:
        print(f"❌ /query connection error: {e}")
        return False

if __name__ == "__main__":
    # Wait for server to start
    print("Waiting 5s for server to start...")
    time.sleep(5)
    
    if check_health() and check_root() and check_query():
        print("\n🎉 ALL APIs are WORKING!")
        sys.exit(0)
    else:
        print("\n⚠️ Some APIs failed.")
        sys.exit(1)
