# src/smart_profile.py
import re
import math

def extract_numbers(text):
    nums = re.findall(r"\d+(?:\.\d+)?", text)
    return [float(n) for n in nums]

# ------------------ PROFILE PARSER ------------------ #

def infer_profile_from_query(query: str):
    """
    Tự động suy luận thông tin từ câu hỏi.
    Nếu thiếu → trả về default an toàn.
    """

    q = query.lower()

    numbers = extract_numbers(q)
    weight = None
    age = None

    # Suy luận cân nặng (kg)
    for n in numbers:
        if 30 <= n <= 200:
            weight = n
            break

    # Suy luận tuổi
    for n in numbers:
        if 10 <= n <= 80:
            age = n
            break

    gender = None
    if "nam" in q:
        gender = "male"
    if "nữ" in q or "nu" in q:
        gender = "female"

    # Giá trị mặc định nếu thiếu
    if weight is None:
        weight = 65  # default VN

    if age is None:
        age = 25

    if gender is None:
        gender = "male"

    return {
        "weight": weight,
        "age": age,
        "gender": gender
    }

# ------------------ BMR / TDEE ------------------ #

def predict_bmr(weight, height, age, gender):
    """
    Không bao giờ lỗi vì toàn bộ biến được đảm bảo != None.
    """
    weight = float(weight)
    height = float(height)
    age = float(age)

    if gender == "male":
        return 88.36 + (13.4 * weight) + (4.8 * height) - (5.7 * age)
    else:
        return 447.6 + (9.2 * weight) + (3.1 * height) - (4.3 * age)

def predict_tdee(bmr, activity="moderate"):
    multipliers = {
        "low": 1.2,
        "light": 1.375,
        "moderate": 1.55,
        "high": 1.725,
        "very_high": 1.9
    }
    return bmr * multipliers.get(activity, 1.55)
