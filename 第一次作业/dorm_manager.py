import json

DATA_FILE = "students.json"

def load_data():
    # 读取students.json或返回空列表
    try:
        with open(DATA_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, list):
                return data
            return []
    except Exception:
        return []

def save_data(stu_list):
    # 保存students.json
    with open(DATA_FILE, "w", encoding="utf-8") as f:
        json.dump(stu_list, f, ensure_ascii=False, indent=2)

def find_by_id(stu_list, sid):
    # 查找学号
    for s in stu_list:
        if s["id"] == sid:
            return s
    return None

def input_non_empty(prompt):
    # 输入任意字符串
    while True:
        s = input(prompt).strip()
        if s:
            return s
        print("Input cannot be empty.")

def input_gender():
    # 输入性别
    while True:
        g = input("Gender (M/F): ").strip()
        if g.lower() == "m":
            return "M"
        if g.lower() == "f":
            return "F"
        print("Only M or F.")


def add_student(stu_list):
    # 添加学生
    print("-- Add Student --")
    while True:
        sid = input_non_empty("ID: ")
        if find_by_id(stu_list, sid):
            print("ID already exists, try another.")
        else:
            break

    name = input_non_empty("Name: ")
    gender = input_gender()
    room = input_non_empty("Room: ")
    phone = input_non_empty("Phone: ")

    stu = {"id": sid, "name": name, "gender": gender, "room": room, "phone": phone}
    stu_list.append(stu)
    print("Added:")
    print_row(stu)

def search_student(stu_list):
    # 根据学号搜索学生
    print("-- Search By ID --")
    sid = input_non_empty("ID: ")
    s = find_by_id(stu_list, sid)
    if s:
        print_header()
        print_row(s)
    else:
        print("ID not found.")

def show_all(stu_list):
    # 显示学生列表
    if not stu_list:
        print("No student records yet.")
        return
    print_header()
    for s in stu_list:
        print_row(s)

def print_header():
    # 对齐输出
    print(f"{'ID':<12} {'Name':<10} {'Gender':<6} {'Room':<10} {'Phone':<15}")
    print("-" * 55)


def print_row(s):
    print(f"{s['id']:<12} {s['name']:<10} {s['gender']:<6} {s['room']:<10} {s['phone']:<15}")


def menu():
    stu_list = load_data()
    while True:
        print("\n===== Dorm Manager =====")
        print("1) Add Student")
        print("2) Search By ID")
        print("3) Show All")
        print("4) Save & Exit")
        choice = input("Choose (1-4): ").strip()

        if choice == "1":
            add_student(stu_list)
        elif choice == "2":
            search_student(stu_list)
        elif choice == "3":
            show_all(stu_list)
        elif choice == "4":
            save_data(stu_list)
            print("Saved to students.json.")
            break
        else:
            print("Invalid choice.")


if __name__ == "__main__":
    menu()

