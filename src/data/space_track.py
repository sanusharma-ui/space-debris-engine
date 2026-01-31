import requests
from getpass import getpass

def space_track_login():
    username = input("Space-Track username/email: ")
    password = getpass("Space-Track password: ")

    session = requests.Session()
    login_url = "https://www.space-track.org/ajaxauth/login"
    payload = {"identity": username, "password": password}

    resp = session.post(login_url, data=payload, timeout=15)
    if resp.status_code != 200 or "You have successfully logged in" not in resp.text:
        raise RuntimeError("Space-Track login failed")

    return session
