import glob
import json

txt_question_2021 = glob.glob("./alltext/alltxt/*.txt")
first_label = []
for txt_question_one in txt_question_2021:
    txt_question_list = open(txt_question_one, "r").readlines()
    for txt_question_str in txt_question_list:
        try:
            txt_question_eval = json.loads(txt_question_str)
            # print(txt_question_eval)
        except:
            # print(txt_question_str)
            txt_question_eval = {'inside': ""}
        # print(txt_question_eval)
        # print(type(txt_question_eval))
        try:
            if isinstance(txt_question_eval['inside'], str):
                txt_question_eval = eval(txt_question_eval['inside'])
        except:
            txt_question_eval = {}
        if isinstance(txt_question_eval, list):
            first_label.extend(txt_question_eval)
from collections import Counter

print(first_label)
count = Counter(first_label)  # 统计词频
print(count)
print(count.most_common(50))
d = []
for i, j in count.most_common(5000):
    d.append({"key": i, "count": j})
import pandas as pd

pd.DataFrame(d).to_csv("most_common.csv")
