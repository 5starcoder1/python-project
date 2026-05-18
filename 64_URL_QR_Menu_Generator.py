# =========================================
# ADVANCED QR CODE GENERATOR IN PYTHON
# =========================================

# FEATURES:
# ✅ Custom QR Color
# ✅ Logo Inside QR
# ✅ Website QR
# ✅ WiFi QR
# ✅ Payment QR
# ✅ PDF QR
# ✅ Dynamic QR Naming
# ✅ Beginner Friendly

# -----------------------------------------
# INSTALL REQUIRED LIBRARIES
# -----------------------------------------
# pip install qrcode[pil] pillow

# -----------------------------------------
# IMPORTS
# -----------------------------------------

import qrcode
from PIL import Image

print("===================================")
print("     ADVANCED QR GENERATOR")
print("===================================")

print("""
1. Website/Menu QR
2. WiFi QR
3. Payment QR
4. PDF QR
""")

choice = input("Select Option: ")

# -----------------------------------------
# GET QR DATA
# -----------------------------------------

if choice == "1":

    data = input("Enter Website/Menu URL: ")

elif choice == "2":

    ssid = input("Enter WiFi Name (SSID): ")
    password = input("Enter WiFi Password: ")

    data = f"WIFI:T:WPA;S:{ssid};P:{password};;"

elif choice == "3":

    upi = input("Enter UPI ID: ")
    name = input("Enter Name: ")
    amount = input("Enter Amount: ")

    data = (
        f"upi://pay?pa={upi}"
        f"&pn={name}"
        f"&am={amount}"
        f"&cu=INR"
    )

elif choice == "4":

    pdf_url = input("Enter PDF URL: ")
    data = pdf_url

else:
    print("❌ Invalid Option")
    exit()

# -----------------------------------------
# QR COLOR
# -----------------------------------------

fill_color = input(
    "Enter QR Color (example: black, blue, red): "
)

background_color = input(
    "Enter Background Color (example: white, yellow): "
)

# -----------------------------------------
# CREATE QR
# -----------------------------------------

qr = qrcode.QRCode(
    version=1,
    error_correction=qrcode.constants.ERROR_CORRECT_H,
    box_size=10,
    border=4
)

qr.add_data(data)

qr.make(fit=True)

img = qr.make_image(
    fill_color=fill_color,
    back_color=background_color
).convert('RGB')

# -----------------------------------------
# ADD LOGO
# -----------------------------------------

add_logo = input("Do you want to add logo? (yes/no): ")

if add_logo.lower() == "yes":

    logo_path = input("Enter logo image path: ")

    try:

        logo = Image.open(logo_path)

        # Resize Logo
        logo_size = 100

        logo = logo.resize((logo_size, logo_size))

        # QR Size
        qr_width, qr_height = img.size

        # Position
        position = (
            (qr_width - logo_size) // 2,
            (qr_height - logo_size) // 2
        )

        # Paste Logo
        img.paste(logo, position)

        print("✅ Logo Added")

    except:
        print("❌ Logo Error")

# -----------------------------------------
# SAVE FILE
# -----------------------------------------

file_name = input(
    "Enter file name to save (without .png): "
)

final_name = file_name + ".png"

img.save(final_name)

print("\n===================================")
print("✅ QR Code Generated Successfully!")
print(f"📁 Saved as: {final_name}")
print("===================================")

# Show QR
img.show()
