import requests

CELESTRAK_URL = "https://celestrak.org/NORAD/elements/gp.php"

def fetch_tle(norad_id: int):
    """
    Fetch TLE for a given NORAD ID from CelesTrak (robust method).
    Uses gp.php endpoint which directly filters by CATNR.
    """
    url = f"{CELESTRAK_URL}?CATNR={norad_id}&FORMAT=TLE"
    resp = requests.get(url, timeout=10)

    if resp.status_code != 200:
        raise RuntimeError("Failed to fetch TLE from CelesTrak")

    lines = [l.strip() for l in resp.text.splitlines() if l.strip()]

    if len(lines) < 2:
        raise ValueError(f"TLE not found for NORAD ID {norad_id}")

    # Some responses include name, some don't
    if lines[0].startswith("1 "):
        name = f"NORAD-{norad_id}"
        tle1, tle2 = lines[0], lines[1]
    else:
        name = lines[0]
        tle1, tle2 = lines[1], lines[2]

    return name, tle1, tle2
