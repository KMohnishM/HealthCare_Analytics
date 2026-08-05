import re
import requests

username = "kmohnishm"
password = "HereisMy2006Bye"

session = requests.Session()

# 1. Login
login_url = "https://physionet.org/login/"
r_get = session.get(login_url, timeout=15)
csrf_token = session.cookies.get("csrftoken")
payload = {
    "username": username,
    "password": password,
    "csrfmiddlewaretoken": csrf_token,
    "next": "/"
}
headers = {
    "Referer": login_url,
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}
session.post(login_url, data=payload, headers=headers, timeout=15)

# 2. Get directory listing of version 2.1.0
cxr_dir_url = "https://physionet.org/files/mimic-cxr-jpg/2.1.0/"
print("GET CXR 2.1.0 folder listing...")
r_dir = session.get(cxr_dir_url, headers={"Referer": "https://physionet.org/"}, timeout=15)
print("Directory Listing Status:", r_dir.status_code)

# Parse links
links = re.findall(r'href="([^"]+)"', r_dir.text)
print("\nLinks found in 2.1.0 directory:")
for link in links:
    if not link.startswith('?') and not link.startswith('/'):
        print("  ", link)

# Also test version 2.0.0 directory listing
cxr_dir_200 = "https://physionet.org/files/mimic-cxr-jpg/2.0.0/"
print("\nGET CXR 2.0.0 folder listing...")
r_dir_200 = session.get(cxr_dir_200, headers={"Referer": "https://physionet.org/"}, timeout=15)
print("Directory Listing 2.0.0 Status:", r_dir_200.status_code)

links_200 = re.findall(r'href="([^"]+)"', r_dir_200.text)
print("\nLinks found in 2.0.0 directory:")
for link in links_200:
    if not link.startswith('?') and not link.startswith('/'):
        print("  ", link)
