import requests

auth = ("kmohnishm", "HereisMy2006Bye")

urls = [
    "https://physionet.org/files/mimic-cxr-jpg/2.1.0/mimic-cxr-2.1.0-metadata.csv.gz",
    "https://physionet.org/files/mimic-cxr-jpg/2.0.0/mimic-cxr-2.0.0-metadata.csv.gz",
    "https://physionet.org/files/mimic-cxr/2.0.0/mimic-cxr-2.0.0-metadata.csv.gz",
]

for url in urls:
    print(f"Testing HEAD on {url}...")
    try:
        r = requests.head(url, auth=auth, timeout=15)
        print(f"  Status: {r.status_code}")
        if r.status_code == 200:
            print("  Headers:", r.headers)
    except Exception as e:
        print(f"  Error: {e}")
