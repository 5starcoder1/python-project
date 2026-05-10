# =========================
# EXPENSE TRACKER IN PYTHON
# =========================

expenses = []

def add_expense():
    name = input("Enter expense name: ")
    amount = float(input("Enter amount: ₹"))

    expense = {
        "name": name,
        "amount": amount
    }

    expenses.append(expense)
    print("✅ Expense added successfully!")

def view_expenses():
    if len(expenses) == 0:
        print("\nNo expenses found!")
        return

    print("\n===== ALL EXPENSES =====")

    total = 0

    for i, expense in enumerate(expenses, start=1):
        print(f"{i}. {expense['name']} - ₹{expense['amount']}")
        total += expense['amount']

    print(f"\n💰 Total Expense: ₹{total}")

def delete_expense():
    view_expenses()

    if len(expenses) == 0:
        return

    try:
        num = int(input("\nEnter expense number to delete: "))
        removed = expenses.pop(num - 1)
        print(f"❌ Deleted: {removed['name']}")
    except:
        print("⚠ Invalid number!")

while True:
    print("\n===== EXPENSE TRACKER =====")
    print("1. Add Expense")
    print("2. View Expenses")
    print("3. Delete Expense")
    print("4. Exit")

    choice = input("Enter choice: ")

    if choice == "1":
        add_expense()

    elif choice == "2":
        view_expenses()

    elif choice == "3":
        delete_expense()

    elif choice == "4":
        print("👋 Exiting Expense Tracker...")
        break

    else:
        print("⚠ Invalid choice! Try again.")
