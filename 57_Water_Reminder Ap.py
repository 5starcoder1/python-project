# =========================
# WATER REMINDER APP PYTHON
# =========================

import time
from plyer import notification

print("===== WATER REMINDER APP =====")

# Time interval in minutes
minutes = int(input("Enter reminder interval (minutes): "))

seconds = minutes * 60

print(f"\n💧 Reminder will appear every {minutes} minute(s)...")

while True:

    # Show Notification
    notification.notify(
        title="💧 Water Reminder",
        message="Time to drink water and stay healthy!",
        timeout=10
    )

    print("💧 Reminder Sent!")

    # Wait
    time.sleep(seconds)
