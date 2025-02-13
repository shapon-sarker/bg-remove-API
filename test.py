import requests

url = "http://127.0.0.1:8000/remove_bg/"
file_path = "input.jpg"

try:
    with open(file_path, "rb") as file:
        files = {"file": file}
        response = requests.post(url, files=files)

    # ✅ Status Code চেক করুন
    print(f"🔍 Status Code: {response.status_code}")

    # ✅ রেসপন্স JSON চেক করুন
    if response.status_code == 200:
        print("✅ Success:", response.json())
    else:
        print("❌ Error:", response.text)

except requests.exceptions.ConnectionError:
    print("❌ Could not connect to the server. Make sure FastAPI is running.")
except requests.exceptions.RequestException as e:
    print(f"❌ Request Error: {e}")
