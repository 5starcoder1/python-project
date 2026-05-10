# =============================
# TYPING SPEED TEST IN PYTHON
# =============================

import time

# Sentence for typing test
sentence = "Python is one of the most popular programming languages."

print("\n===== TYPING SPEED TEST =====")
print("\nType the following sentence:\n")
print(sentence)

input("\nPress ENTER when you are ready...")

# Start timer
start_time = time.time()

# User input
typed_text = input("\nStart Typing:\n")

# End timer
end_time = time.time()

# Time taken
time_taken = end_time - start_time

# Count words
word_count = len(typed_text.split())

# Calculate WPM
wpm = (word_count / time_taken) * 60

# Accuracy check
correct_chars = 0

for i in range(min(len(sentence), len(typed_text))):
    if sentence[i] == typed_text[i]:
        correct_chars += 1

accuracy = (correct_chars / len(sentence)) * 100

# Results
print("\n===== RESULT =====")
print(f"⏱ Time Taken: {round(time_taken, 2)} seconds")
print(f"⌨ Typing Speed: {round(wpm, 2)} WPM")
print(f"🎯 Accuracy: {round(accuracy, 2)}%")

if accuracy == 100:
    print("🔥 Perfect Typing!")
elif accuracy >= 80:
    print("✅ Good Job!")
else:
    print("⚠ Practice More!")
