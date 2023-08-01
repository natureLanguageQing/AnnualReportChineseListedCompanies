import glob
import json

txt_question_2021 = glob.glob("./alltext/alltxt/*.txt")
first_label = []
form_list = {}
for txt_question_one in txt_question_2021:
    form_str = ""
    txt_question_list = open(txt_question_one, "r").readlines()
    error_index = 0
    for txt_question_index in range(len(txt_question_list)):
        txt_question_str = txt_question_list[txt_question_index]
        try:
            txt_question_eval = json.loads(txt_question_str)
            # print(txt_question_eval)
        except:
            # print(txt_question_str)
            txt_question_eval = {}
        if 'inside' in txt_question_eval:
            try:
                inside_eval = eval(txt_question_eval['inside'])
                # print(inside_eval)
            except:
                inside_eval = ""

        else:
            inside_eval = ""
        # if isinstance(inside_eval, dict):
        #     form_str = ""
        #     continue
        # print(inside_eval)
        # print(type(inside_eval))
        if inside_eval:
            # if "年 度 报 告" in inside_eval:
            #     continue
            if isinstance(inside_eval, list):
                if len(form_str) == 0:
                    # form_str += " ".join(json.loads(txt_question_list[txt_question_index - 1])['inside']) + "\n"
                    append_index = txt_question_index - 5
                    append = 0

                    # form_str_add = " ".join(json.loads(txt_question_list[txt_question_index - 2])['inside']) + "\n"
                    while append < 2:
                        try:
                            form_str_add = json.loads(txt_question_list[append_index])['inside'] + "\n"
                        except:
                            form_str_add = ""
                            append = 2
                        if "年 度 报 告" not in form_str_add and "年年度报告" not in form_str_add:
                            form_str += form_str_add
                            append += 1
                        else:
                            append_index += 1
                            form_str_add = json.loads(txt_question_list[append_index])['inside'] + "\n"

                # first_label.extend(txt_question_eval)
                inside_eval_str = []
                for inside_eval_inner in inside_eval:
                    inside_eval_str.append(str(inside_eval_inner))
                try:
                    form_str += "|" + "|".join(inside_eval_str) + "|\n"
                except:
                    print(inside_eval_str)
            else:
                error_index += 1
                if error_index == 3:
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

json.dump(form_list, open("form_list.json", "w"), ensure_ascii=False)
