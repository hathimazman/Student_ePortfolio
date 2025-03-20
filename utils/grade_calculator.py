def grade_calculate(data, grade_cutoffs=None):
   
    result_data = data.copy()

    grade = ""
   
    if result_data >= 80:
        grade = "A"
    elif result_data >= 75:
        grade = "A-"
    elif result_data >= 70:
        grade = "B+"
    elif result_data >= 65:
        grade = "B"
    elif result_data >= 60:
        grade = "B-"
    elif result_data >= 55:
        grade = "C+"
    elif result_data >= 50:
        grade = "C"
    elif result_data >= 45:
        grade = "C-"
    elif result_data >= 40:
        grade = "D+"
    elif result_data >= 35:
        grade = "D"
    else:
        grade = "E"
    
    return grade