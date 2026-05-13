# ===========================
# AUTO SHUTDOWN TOOL PYTHON
# ===========================

import os
import time

print("===== AUTO SHUTDOWN TOOL =====")

# User Input
minutes = int(input("Enter time in minutes for shutdown: "))

seconds = minutes * 60

print(f"\n⏳ Your PC will shutdown in {minutes} minute(s)...")

# Countdown
for i in range(seconds, 0, -1):
    mins = i // 60
    secs = i % 60

    print(f"Time Left: {mins:02d}:{secs:02d}", end="\r")

    time.sleep(1)

# Shutdown Command
os.system("shutdown /s /t 1")

"""
❌ Shutdown Cancel Karne Ka Command

Agar shutdown cancel karna ho to terminal me ye command run karo:

shutdown /a
"""
