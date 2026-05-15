# =========================
# URL SHORTENER IN PYTHON
# =========================

import pyshorteners

print("===== URL SHORTENER =====")

# Long URL Input
long_url = input("Enter Long URL: ")

# Create Shortener Object
shortener = pyshorteners.Shortener()

# Shorten URL
short_url = shortener.tinyurl.short(long_url)

# Output
print("\n✅ Short URL Generated Successfully!")
print("🔗 Short URL:", short_url)
