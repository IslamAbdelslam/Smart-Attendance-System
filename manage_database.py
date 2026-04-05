#--------------------------------------------------------------------------------
# Manage date base 
# it uses to manage the database (Delete a name , clear all the database ,or quit)
#---------------------------------------------------------------------------------
import pickle
import os
import numpy as np

FILE_PATH = r"C:\Users\Administrator\Documents\Smart Attendence System\Database\database.pkl"

def manage_db():
    if not os.path.exists(FILE_PATH):
        print(f"❌ Error: {FILE_PATH} not found in this folder.")
        return

    # 1. Load the database
    with open(FILE_PATH, "rb") as f:
        database = pickle.load(f)

    while True:
        print("\n" + "="*30)
        print("  DATABASE MANAGER")
        print("="*30)
        print(f"Total Enrolled: {len(database)}")
        print("-" * 30)
        
        # 2. List all names
        if not database:
            print("Database is empty.")
        else:
            names = sorted(database.keys())
            for i, name in enumerate(names, 1):
                print(f"{i}. {name}")

        print("-" * 30)
        print("Options: [d] Delete a name | [c] Clear All | [q] Quit")
        choice = input("Select an option: ").lower().strip()

        # 3. Handle Deletion
        if choice == 'd':
            target = input("Enter the EXACT name to delete: ").strip()
            if target in database:
                del database[target]
                with open(FILE_PATH, "wb") as f:
                    pickle.dump(database, f)
                print(f"✅ Successfully deleted '{target}'.")
            else:
                print(f"❌ Name '{target}' not found. (Check spaces/capitalization)")
        
        # 4. Handle Clear All
        elif choice == 'c':
            confirm = input("⚠️ Are you sure you want to delete EVERYONE? (y/n): ")
            if confirm.lower() == 'y':
                database = {}
                with open(FILE_PATH, "wb") as f:
                    pickle.dump(database, f)
                print("✅ Database cleared.")

        elif choice == 'q':
            print("Exiting manager.")
            break

if __name__ == "__main__":
    manage_db()