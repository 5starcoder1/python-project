# =========================
# YOUTUBE VIDEO DOWNLOADER
# =========================

from pytube import YouTube

print("===== YOUTUBE VIDEO DOWNLOADER =====")

# Video URL
url = input("Enter YouTube Video URL: ")

try:
    # Create YouTube Object
    yt = YouTube(url)

    print(f"\n🎬 Title: {yt.title}")
    print(f"👀 Views: {yt.views}")

    # Highest Resolution Stream
    stream = yt.streams.get_highest_resolution()

    print("\n⏳ Downloading...")

    # Download Video
    stream.download()

    print("✅ Video Downloaded Successfully!")

except Exception as e:
    print("❌ Error:", e)
