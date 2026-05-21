# =================================================
# ADVANCED AI IMAGE GENERATOR USING PYTHON
# =================================================

# FEATURES:
# ✅ AI Text To Image
# ✅ Multiple Image Styles
# ✅ HD Image Generation
# ✅ Save Generated Images
# ✅ GUI Interface
# ✅ Dark Theme
# ✅ Beginner Friendly
# ✅ Uses Hugging Face AI API

# -------------------------------------------------
# INSTALL REQUIRED LIBRARIES
# -------------------------------------------------
# pip install customtkinter requests pillow

# -------------------------------------------------
# IMPORTS
# -------------------------------------------------

import customtkinter as ctk
import requests
from PIL import Image, ImageTk
from io import BytesIO

# -------------------------------------------------
# HUGGING FACE API
# -------------------------------------------------

API_URL = (
    "https://api-inference.huggingface.co/models/"
    "stabilityai/stable-diffusion-xl-base-1.0"
)

# 🔥 ENTER YOUR HUGGING FACE TOKEN
HEADERS = {
    "Authorization": "Bearer YOUR_HUGGINGFACE_API_KEY"
}

# -------------------------------------------------
# APP SETTINGS
# -------------------------------------------------

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

app = ctk.CTk()

app.title("🔥 AI IMAGE GENERATOR")

app.geometry("900x700")

# -------------------------------------------------
# GENERATE IMAGE FUNCTION
# -------------------------------------------------

def generate_image():

    prompt = prompt_entry.get()

    style = style_option.get()

    final_prompt = f"{prompt}, {style}"

    status_label.configure(
        text="⏳ Generating Image..."
    )

    payload = {
        "inputs": final_prompt
    }

    try:

        response = requests.post(
            API_URL,
            headers=HEADERS,
            json=payload
        )

        image_bytes = response.content

        image = Image.open(BytesIO(image_bytes))

        # Resize Image
        image = image.resize((500, 500))

        img = ImageTk.PhotoImage(image)

        image_label.configure(image=img)

        image_label.image = img

        # Save Image
        image.save("generated_ai_image.png")

        status_label.configure(
            text="✅ Image Generated & Saved"
        )

    except Exception as e:

        status_label.configure(
            text=f"❌ Error: {e}"
        )

# -------------------------------------------------
# UI TITLE
# -------------------------------------------------

title = ctk.CTkLabel(
    app,
    text="🔥 AI IMAGE GENERATOR",
    font=("Arial", 30, "bold")
)

title.pack(pady=20)

# -------------------------------------------------
# PROMPT INPUT
# -------------------------------------------------

prompt_entry = ctk.CTkEntry(
    app,
    width=600,
    height=45,
    placeholder_text="Enter your image prompt..."
)

prompt_entry.pack(pady=10)

# -------------------------------------------------
# STYLE OPTIONS
# -------------------------------------------------

style_option = ctk.CTkOptionMenu(
    app,
    values=[
        "Realistic",
        "Anime",
        "Cyberpunk",
        "3D Art",
        "Fantasy",
        "Cartoon",
        "Ultra HD",
        "Digital Art"
    ]
)

style_option.pack(pady=10)

# -------------------------------------------------
# GENERATE BUTTON
# -------------------------------------------------

generate_btn = ctk.CTkButton(
    app,
    text="🚀 Generate Image",
    width=250,
    height=45,
    command=generate_image
)

generate_btn.pack(pady=20)

# -------------------------------------------------
# STATUS LABEL
# -------------------------------------------------

status_label = ctk.CTkLabel(
    app,
    text="",
    font=("Arial", 16)
)

status_label.pack(pady=10)

# -------------------------------------------------
# IMAGE DISPLAY
# -------------------------------------------------

image_label = ctk.CTkLabel(
    app,
    text=""
)

image_label.pack(pady=20)

# -------------------------------------------------
# RUN APP
# -------------------------------------------------

app.mainloop()
