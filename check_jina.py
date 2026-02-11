import os
import requests

print(f"Provider: {os.getenv('WEB_SEARCH_PROVIDER')}")
print(f"Jina Key Set: {bool(os.getenv('JINA_API_KEY'))}")

if os.getenv('JINA_API_KEY'):
    headers = {"Authorization": f"Bearer {os.getenv('JINA_API_KEY')}", "X-Respond-With": "no-content"}
    try:
        resp = requests.get("https://s.jina.ai/?q=Ali%20Khan", headers=headers, timeout=10)
        print("--- Jina Response Start ---")
        print(resp.text[:500])
        print("--- Jina Response End ---")
    except Exception as e:
        print(f"Error: {e}")
