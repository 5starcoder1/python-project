# =========================
# MEME GENERATOR IN PYTHON
# =========================

from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont

print("===== MEME GENERATOR =====")

# Load Image
image_path = input("Enter image path: ")

img = Image.open(image_path)

# Text Input
top_text = input("Enter TOP text: ")
bottom_text = input("Enter BOTTOM text: ")

# Draw Object
draw = ImageDraw.Draw(img)

# Font
font = ImageFont.truetype("arial.ttf", 40)

# Image Size
width, height = img.size

# Top Text Position
draw.text(
    (width / 4, 20),
    top_text,
    font=font,
    fill="white"
)

# Bottom Text Position
draw.text(
    (width / 4, height - 80),
    bottom_text,
    font=font,
    fill="white"
)

# Save Meme
img.save("generated_meme.jpg")

print("\n✅ Meme Generated Successfully!")
print("📁 Saved as: generated_meme.jpg")

# Show Meme
img.show()
