# =========================
# IPL LIVE SCORE APP PYTHON
# =========================

import requests
import time

API_KEY = "YOUR_API_KEY"

print("===== IPL LIVE SCORE APP =====")

while True:

    url = f"https://api.cricapi.com/v1/currentMatches?apikey={API_KEY}&offset=0"

    response = requests.get(url)

    data = response.json()

    print("\n🏏 LIVE IPL MATCHES 🏏\n")

    matches = data.get("data", [])

    found = False

    for match in matches:

        # Check IPL Match
        if "IPL" in match.get("series", ""):

            found = True

            print(f"📌 Match: {match.get('name')}")
            print(f"📍 Status: {match.get('status')}")

            score = match.get("score", [])

            for inning in score:
                print(
                    f"🏏 {inning['inning']} : "
                    f"{inning['r']}/{inning['w']} "
                    f"({inning['o']} Overs)"
                )

            print("-" * 40)

    if not found:
        print("❌ No Live IPL Matches Found")

    # Refresh every 30 seconds
    print("\n🔄 Refreshing in 30 seconds...\n")

    time.sleep(30)
