# =================================
# WHATSAPP AUTO SENDER PYTHON
# =================================

import pywhatkit
import time


print("================================")
print("   WHATSAPP AUTO SENDER")
print("================================")


# Receiver Number
number = input(
    "Enter WhatsApp Number with country code (+91): "
)


# Message
message = input(
    "Enter Message: "
)


# Time
hour = int(input("Enter Hour (24 hour format): "))
minute = int(input("Enter Minute: "))


print("\n⏳ Scheduling Message...")


try:

    pywhatkit.sendwhatmsg(
        number,
        message,
        hour,
        minute
    )


    print(
        "✅ Message Scheduled Successfully"
    )


except Exception as e:

    print(
        "❌ Error:",
        e
    )


time.sleep(5)
