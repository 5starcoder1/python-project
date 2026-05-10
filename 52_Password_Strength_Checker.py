# ===================================
# PASSWORD STRENGTH CHECKER IN PYTHON
# ===================================

import re

print("===== PASSWORD STRENGTH CHECKER =====")

password = input("Enter your password: ")

strength = 0

# Length Check
if len(password) >= 8:
    strength += 1

# Uppercase Check
if re.search(r"[A-Z]", password):
    strength += 1

# Lowercase Check
if re.search(r"[a-z]", password):
    strength += 1

# Number Check
if re.search(r"[0-9]", password):
    strength += 1

# Special Character Check
if re.search(r"[!@#$%^&*(),.?\":{}|<>]", password):
    strength += 1

# Result
print("\n===== RESULT =====")

if strength == 5:
    print("🔥 Very Strong Password")
elif strength == 4:
    print("✅ Strong Password")
elif strength == 3:
    print("⚠ Medium Password")
else:
    print("❌ Weak Password")

# Suggestions
print("\n===== SUGGESTIONS =====")

if len(password) < 8:
    print("- Add at least 8 characters")

if not re.search(r"[A-Z]", password):
    print("- Add uppercase letters")

if not re.search(r"[a-z]", password):
    print("- Add lowercase letters")

if not re.search(r"[0-9]", password):
    print("- Add numbers")

if not re.search(r"[!@#$%^&*(),.?\":{}|<>]", password):
    print("- Add special characters")
