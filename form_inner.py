import glob
import json

txt_question_2021 = glob.glob("./alltext/alltxt/*.txt")
first_label = []
form_list = {}
import json

keyword_list = json.load(open("keyword_list.json", "r"))
in_stock_name = open("in_stock_name.json", "r")
in_stock_name = json.load(in_stock_name)
# wenjianmingcheng duiyingwenjianmingchengzhongchuxianlewentiguanjiancidebiaoge
error_index = 0
for txt_question_one in txt_question_2021:
    form_str = ""
    for in_stock_key, in_stock_value in in_stock_name.items():
        if in_stock_key in txt_question_one:
            for in_stock_one in in_stock_value:
                if in_stock_one in txt_question_one:
                    txt_question_list = open(txt_question_one, "r").readlines()
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
                                error_index += 1
                                if error_index >= 2:
                                    if len(form_str):
                                        for keyword_one in keyword_list:
                                            if keyword_one in form_str:
                                                if txt_question_one not in form_list:
                                                    form_list[txt_question_one] = [form_str]
                                                else:
                                                    form_list[txt_question_one].append(form_str)
                                        form_str = ""
                                        error_index = 0
                # first_label.append(form_list)
                # form_list = {}
for form_key, form_value in form_list.items():
    form_list[form_key] = list(set(form_value))

import json

json.dump(form_list, open("form_list_keywords.json", "w"), ensure_ascii=False)
