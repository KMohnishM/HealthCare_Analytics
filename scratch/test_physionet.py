import requests

auth = ("kmohnishm", "HereisMy2006Bye")
url_ecg_real = "https://physionet.org/files/mimic-iv-ecg/1.0/files/p1000/p10000032/s40689238/40689238.hea"

print("Testing actual ECG record file from record_list.csv with credentials...")
try:
    r = requests.head(url_ecg_real, auth=auth, timeout=15)
    print("ECG Real HEA Status:", r.status_code)
    print("Headers:", r.headers)
except Exception as e:
    print("Error:", e)
