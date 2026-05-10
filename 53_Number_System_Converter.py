# ==================================
# NUMBER SYSTEM CONVERTER IN PYTHON
# ==================================

print("===== NUMBER SYSTEM CONVERTER =====")

number = int(input("Enter a decimal number: "))

# Convert Number
binary = bin(number)
octal = oct(number)
hexadecimal = hex(number)

# Display Result
print("\n===== CONVERSION RESULT =====")
print(f"Decimal     : {number}")
print(f"Binary      : {binary}")
print(f"Octal       : {octal}")
print(f"Hexadecimal : {hexadecimal}")
