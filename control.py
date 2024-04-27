import requests

BASE_URL = 'http://localhost:6969'

# Function to get context by resourceid
def get_context_by_resourceid(resource_id):
    url = f"{BASE_URL}/get_context/{resource_id}"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Error: {response.status_code}")

# Function to set context by resourceid
def set_context_by_resourceid(resource_id, new_context):
    url = f"{BASE_URL}/set_context/{resource_id}"
    payload = {'new_context': new_context}
    response = requests.put(url, json=payload)
    if response.status_code == 200:
        print("Context updated successfully")
    else:
        print(f"Error: {response.status_code}")

def get_style_by_resourceid(resource_id):
    url = f"{BASE_URL}/get_style/{resource_id}"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Error: {response.status_code}")

# Function to set style by resourceid
def set_style_by_resourceid(resource_id, new_style):
    url = f"{BASE_URL}/set_style/{resource_id}"
    payload = {'new_style': new_style}
    response = requests.put(url, json=payload)
    if response.status_code == 200:
        print("Style updated successfully")
    else:
        print(f"Error: {response.status_code}")