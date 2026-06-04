# =====================================
# AI JOB APPLICATION ASSISTANT BOT
# =====================================

import pandas as pd
import os
from datetime import datetime


FILE = "job_tracker.csv"


# -------------------------------
# Create Database
# -------------------------------

def create_file():

    if not os.path.exists(FILE):

        df = pd.DataFrame(
            columns=[
                "Company",
                "Role",
                "Link",
                "Status",
                "Date"
            ]
        )

        df.to_csv(FILE, index=False)



# -------------------------------
# Add Job
# -------------------------------

def add_job():

    company = input("Company Name: ")

    role = input("Job Role: ")

    link = input("Job Link: ")


    data = pd.read_csv(FILE)


    new_job = {
        "Company": company,
        "Role": role,
        "Link": link,
        "Status": "Applied",
        "Date": datetime.now()
    }


    data.loc[len(data)] = new_job


    data.to_csv(
        FILE,
        index=False
    )


    print("✅ Job Added")



# -------------------------------
# View Jobs
# -------------------------------

def show_jobs():

    data = pd.read_csv(FILE)

    print("\n===== JOB TRACKER =====\n")

    print(data)



# -------------------------------
# Update Status
# -------------------------------

def update_status():

    data = pd.read_csv(FILE)


    show_jobs()


    index = int(
        input("\nEnter job number: ")
    )


    status = input(
        "New Status (Interview/Rejected/Selected): "
    )


    data.loc[index, "Status"] = status


    data.to_csv(
        FILE,
        index=False
    )


    print("✅ Updated")



# -------------------------------
# Resume Match Checker
# -------------------------------

def resume_match():

    resume = input(
        "Enter Resume Skills: "
    ).lower()


    job = input(
        "Enter Job Requirement: "
    ).lower()


    resume_words = set(
        resume.split()
    )


    job_words = set(
        job.split()
    )


    score = (
        len(resume_words & job_words)
        /
        len(job_words)
    ) * 100


    print(
        f"\n🎯 Match Score: {round(score,2)}%"
    )



# -------------------------------
# MAIN MENU
# -------------------------------

create_file()


while True:

    print("""
=========================
 AI JOB ASSISTANT BOT
=========================

1. Add Job
2. View Jobs
3. Update Status
4. Resume Match Score
5. Exit

""")


    choice=input("Choose: ")


    if choice=="1":

        add_job()


    elif choice=="2":

        show_jobs()


    elif choice=="3":

        update_status()


    elif choice=="4":

        resume_match()


    elif choice=="5":

        print("Bye 👋")
        break


    else:

        print("Invalid Choice")
