# =========================
# EMOJI CONVERTER IN PYTHON
# =========================

print("===== EMOJI CONVERTER =====")

# Emoji Dictionary
emoji_dict = {
    ":)": "😊",
    ":(": "😢",
    ":D": "😄",
    "<3": "❤️",
    ":P": "😜",
    ";)": "😉",
    ":o": "😲",
    ":/": "😕",
    "B)": "😎"
}

# User Input
text = input("\nEnter your message: ")

# Convert Emojis
words = text.split()

converted_text = ""

for word in words:
    converted_text += emoji_dict.get(word, word) + " "

# Output
print("\n===== CONVERTED TEXT =====")
print(converted_text)
