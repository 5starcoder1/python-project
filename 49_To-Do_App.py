# =========================
# TO-DO APP IN PYTHON
# =========================

tasks = []

def show_tasks():
    if len(tasks) == 0:
        print("\nNo tasks available!")
    else:
        print("\nYour Tasks:")
        for i, task in enumerate(tasks, start=1):
            print(f"{i}. {task}")

while True:
    print("\n===== TO-DO APP =====")
    print("1. Add Task")
    print("2. View Tasks")
    print("3. Remove Task")
    print("4. Exit")

    choice = input("Enter your choice: ")

    # Add Task
    if choice == "1":
        task = input("Enter task: ")
        tasks.append(task)
        print("✅ Task added successfully!")

    # View Tasks
    elif choice == "2":
        show_tasks()

    # Remove Task
    elif choice == "3":
        show_tasks()

        if len(tasks) != 0:
            try:
                task_num = int(input("Enter task number to remove: "))
                removed = tasks.pop(task_num - 1)
                print(f"❌ Removed task: {removed}")
            except:
                print("Invalid task number!")

    # Exit
    elif choice == "4":
        print("👋 Exiting To-Do App...")
        break

    else:
        print("⚠ Invalid choice! Try again.")
