import glob
import json

txt_question_2021 = glob.glob("./alltext/alltxt/*.txt")
first_label = []
form_list = {}
for txt_question_one in txt_question_2021:
    form_str = ""
    txt_question_list = open(txt_question_one, "r").readlines()
    for txt_question_index in range(len(txt_question_list)):
        txt_question_str = txt_question_list[txt_question_index]
        try:
            txt_question_eval = json.loads(txt_question_str)
            # print(txt_question_eval)
        except:
            # print(txt_question_str)
            txt_question_eval = ""
        if 'inside' in txt_question_eval:
            try:
                inside_eval = eval(txt_question_eval['inside'])
                # print(txt_question_eval['inside'])
            except:
                inside_eval = ""

        else:
            inside_eval = ""
        if isinstance(inside_eval, dict):
            form_str = ""
            continue
        if isinstance(inside_eval, str):
            if len(inside_eval):
                if txt_question_one in form_list:
                    form_list[txt_question_one].append(inside_eval)
                else:
                    form_list[txt_question_one] = [inside_eval]

    # first_label.append(form_list)
    # form_list = {}
import json

json.dump(form_list, open("text_list.json", "w"), ensure_ascii=False)
