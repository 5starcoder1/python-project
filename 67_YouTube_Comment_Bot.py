# =====================================
# YOUTUBE COMMENT AI ASSISTANT BOT
# =====================================

from googleapiclient.discovery import build
import pandas as pd
from datetime import datetime


# ===============================
# YOUTUBE API KEY
# ===============================

API_KEY = "YOUR_YOUTUBE_API_KEY"


youtube = build(
    "youtube",
    "v3",
    developerKey=API_KEY
)


# ===============================
# GET VIDEO ID
# ===============================

def get_video_id(url):

    if "v=" in url:
        return url.split("v=")[1].split("&")[0]

    return url



# ===============================
# FETCH COMMENTS
# ===============================

def get_comments(video_id):

    comments = []


    request = youtube.commentThreads().list(
        part="snippet",
        videoId=video_id,
        maxResults=50
    )


    response = request.execute()


    for item in response["items"]:

        comment = item["snippet"][
            "topLevelComment"
        ]["snippet"]


        comments.append(
            {
                "author":
                comment["authorDisplayName"],

                "comment":
                comment["textDisplay"],

                "likes":
                comment["likeCount"],

                "date":
                comment["publishedAt"]
            }
        )


    return comments



# ===============================
# REPLY SUGGESTION
# ===============================

def generate_reply(comment):

    text = comment.lower()


    if "nice" in text or "good" in text:

        return "Thank you so much ❤️"

    elif "how" in text:

        return "Thanks for asking! I will explain it in detail."

    elif "help" in text:

        return "Sure, I will help you 👍"


    else:

        return "Thank you for watching 😊"



# ===============================
# SAVE DATA
# ===============================

def save_comments(data):

    df = pd.DataFrame(data)

    df.to_csv(
        "youtube_comments.csv",
        index=False
    )

    print(
        "✅ Comments saved"
    )



# ===============================
# MAIN PROGRAM
# ===============================


print(
"""
=============================
 YOUTUBE COMMENT ASSISTANT
=============================
"""
)


url = input(
    "Enter YouTube Video URL: "
)


video_id = get_video_id(url)



try:

    comments = get_comments(video_id)


    for c in comments:

        c["reply_suggestion"] = generate_reply(
            c["comment"]
        )


    save_comments(comments)


    print("\n===== COMMENTS =====")


    for c in comments:

        print(
        f"""
👤 {c['author']}
💬 {c['comment']}
🤖 Suggestion:
{c['reply_suggestion']}
---------------------
"""
        )


except Exception as e:

    print(
        "❌ Error:",
        e
    )
