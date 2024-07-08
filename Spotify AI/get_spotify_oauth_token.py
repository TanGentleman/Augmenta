# This file updates the SPOTIFY_OATH_TOKEN in the .env file
import urllib.parse
from base64 import b64encode
from os import environ
from requests import post
from dotenv import load_dotenv

load_dotenv()

CLIENT_ID = environ.get("SPOTIFY_CLIENT_ID")
CLIENT_SECRET = environ.get("SPOTIFY_CLIENT_SECRET")
if not CLIENT_ID or not CLIENT_SECRET:
    raise Exception("No Spotify client ID or secret found")

redirect_uri = 'https://localhost:3000'
scope = 'user-library-modify user-library-read user-read-email user-read-private'
state = "W0ZZFN8FRxMphLux"
code = "retracted_code"
# Generate a random string for the state parameter
URL = "Custom_url"
# TODO: Write code for getting custom link
# Prerequisite step:
# 1. Sign into spotify in browser
# 2. Visit your custom URL in browser
# 3. Authorize the app
# 4. Copy the value on the right of code= from the redirected url
# 5. Paste that string as the value for the code variable
# You can then run this script!

# Construct the authorization URL
params = {
    'response_type': 'code',
    'client_id': CLIENT_ID,
    'scope': scope,
    'redirect_uri': redirect_uri,
    'state': state
}

auth_url = 'https://accounts.spotify.com/authorize?' + \
    urllib.parse.urlencode(params)

print(auth_url)

response = post(
    "https://accounts.spotify.com/api/token",
    data={
        "grant_type": "authorization_code",
        "code": code,
        "redirect_uri": redirect_uri},
    headers={
        "Authorization": "Basic " +
        b64encode(
            f"{CLIENT_ID}:{CLIENT_SECRET}".encode()).decode(),
        "Content-Type": "application/x-www-form-urlencoded"})
# Parse output
print(response.json())
response_token = response.json()["access_token"]
assert response_token, "No access token found"
with open(".env", "r+") as f:
    lines = f.readlines()
    for i, line in enumerate(lines):
        if line.startswith("SPOTIFY_AUTH_TOKEN"):
            lines[i] = f'SPOTIFY_AUTH_TOKEN="{response_token}"\n'
            break
    else:
        lines.append(f'SPOTIFY_AUTH_TOKEN="{response_token}"\n')
    f.seek(0)
    f.writelines(lines)
print("Token written to .env file")
