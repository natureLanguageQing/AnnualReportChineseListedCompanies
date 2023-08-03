import glob
import os.path

txt_question_2021 = glob.glob("./alltext/alltxt/*.txt")
first_label = []
form_list = {}
text_list = {}
import json

keyword_list = json.load(open("keyword_list.json", "r"))
in_stock_name = open("in_stock_name.json", "r")

in_stock_name = json.load(in_stock_name)

key_file = json.load(open("key_filename.json", "r"))
print(len(key_file))
key_file = list(set(key_file))
print(len(key_file))

# wenjianmingcheng duiyingwenjianmingchengzhongchuxianlewentiguanjiancidebiaoge
error_index = 0
for txt_question_one in txt_question_2021:
    form_str = ""
    for key_file_one in key_file:
        txt_question_one = os.path.join("./alltext/alltxt", key_file_one)
        txt_question_list = open(txt_question_one, "r").readlines()
        text_list[txt_question_one] = []
        form_list[txt_question_one] = []
        for txt_question_index in range(len(txt_question_list)):
            txt_question_str = txt_question_list[txt_question_index]
            try:
                txt_question_eval = json.loads(txt_question_str)
            except:
                txt_question_eval = {}
            if 'inside' in txt_question_eval:
                try:
                    inside_eval = eval(txt_question_eval['inside'])
                except:
                    inside_eval = ""

            else:
                inside_eval = ""
            if inside_eval:
                if isinstance(inside_eval, list):
                    # first_label.extend(txt_question_eval)
                    inside_eval_str = []
                    for inside_eval_inner in inside_eval:
                        inside_eval_str.append(str(inside_eval_inner))
                    try:
                        if len(inside_eval_str) > 1:
                            form_str += "|" + "|".join(inside_eval_str) + "|\n"
                    except:
                        print(inside_eval_str)
                else:
                    if isinstance(inside_eval, str):
                        text_list[txt_question_one].append(inside_eval)
                    error_index += 1
                    if error_index >= 2:
                        if len(form_str):
                            # for keyword_one in keyword_list:
                            #     if keyword_one in form_str:
                            form_list[txt_question_one].append(form_str)
                            form_str = ""
                            error_index = 0
# first_label.append(form_list)
# form_list = {}
for form_key, form_value in form_list.items():
    form_list[form_key] = list(set(form_value))

import json

json.dump(form_list, open("form_list_keywords.json", "w"), ensure_ascii=False)
json.dump(text_list, open("text_list_keywords.json", "w"), ensure_ascii=False)
