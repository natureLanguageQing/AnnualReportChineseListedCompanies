import json

test_questions = open("test_questions_keyword.jsonl").readlines()
question_2020 = []
for test_question in test_questions:
    print(test_question)
    question = json.loads(test_question)
    # if "2020" in question:
    question_2020.append(question)

stock_mapping = json.load(open("stock_in_question.json", "r"))
stock_name = []
for stock_mapping_one in stock_mapping:
    if isinstance(stock_mapping_one, str):
        stock_name.append(stock_mapping_one)
print(stock_name)
stock_mapping = {}
file_form = {}
first_label = json.load(open("form_list_keywords.json", "r"))
for stock_name_one in stock_name:
    for file_name, file_message in first_label.items():
        if stock_name_one in file_name:
            if stock_name_one in stock_mapping:
                stock_mapping[stock_name_one].append(file_name)
            else:
                stock_mapping[stock_name_one] = [file_name]
from transformers import AutoTokenizer, AutoModel
import json
import torch

DEVICE = "cuda"
DEVICE_ID = "0"
CUDA_DEVICE = f"{DEVICE}:{DEVICE_ID}" if DEVICE_ID else DEVICE


def torch_gc():
    if torch.cuda.is_available():
        with torch.cuda.device(CUDA_DEVICE):
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()


import re


def re_chinese(a_str):
    a_list = re.findall(r'[^\x00-\xff]', a_str)
    return "".join(a_list)


if __name__ == '__main__':
    tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm2-6b", trust_remote_code=True)
    # model = AutoModel.from_pretrained("THUDM/chatglm2-6b", trust_remote_code=True, device='cuda')
    model = AutoModel.from_pretrained("THUDM/chatglm2-6b", trust_remote_code=True)
    # 多显卡支持，使用下面两行代替上面一行，将num_gpus改为你实际的显卡数量
    # from utils import load_model_on_gpus
    # model = load_model_on_gpus("THUDM/chatglm2-6b", num_gpus=2)
    # model = model.eval()
    from fastllm_pytools import llm

    model = llm.from_hf(model, tokenizer, dtype="float16")
    for question_one in question_2020:
        all_texts = []
        year_prefix = []
        # question_one = json.loads(question_one)
        if "2019年" in question_one['question']:
            year_prefix.append("2019年")
        if "2020年" in question_one['question']:
            year_prefix.append("2020年")
        if "2021年" in question_one['question']:
            year_prefix.append("2021年")
        # jingzhunpipeibiaoge
        if "keyword" in question_one:
            print(question_one)
            for stock_name_one in stock_name:
                if stock_name_one in question_one['question']:
                    if stock_name_one in stock_mapping:
                        rows = []
                        for stock_form in stock_mapping[stock_name_one]:
                            if year_prefix:
                                for year_prefix_one in year_prefix:
                                    if year_prefix_one in stock_form:
                                        # print(question_one)
                                        for question_one_keyword in question_one['keyword']:
                                            for one in first_label[stock_form]:
                                                if question_one_keyword in one:
                                                    rows.append(one)
                            else:
                                for question_one_keyword in question_one['keyword']:
                                    for one in first_label[stock_form]:
                                        if question_one_keyword in one:
                                            rows.append(one)

                            if len(rows) != 0:
                                rows = list(set(rows))
                                message = "\n".join(rows)
                                message = message[:1024]
                                query = question_one['question']
                                for year_prefix_one in year_prefix:
                                    query.replace(year_prefix_one, "")
                                for stock_name_one in stock_name:
                                    query.replace(stock_name_one, "")

                                all_texts = [
                                    "[Round 0]\n 根据以下几个表格:" + message + " 解决问题：" + query
                                    + "    \n答：",
                                ]
                                # print(all_texts)

        if len(all_texts) == 0:
            all_texts = [
                "[Round 0]\n问：" + question_one['question'] + "    \n答：",
            ]
        response, history = model.chat(tokenizer,
                                       all_texts[0],
                                       history=[],
                                       max_length=2048 + 1024,
                                       top_p=0.7,
                                       temperature=0.95)
        print(all_texts[0])
        print(response)
        question_one['answer'] = response
d = []
for i in question_2020:
    d.append(json.dumps(i, ensure_ascii=False))
open("search_form_text_gen_chinese_without_name.py.json", "w").write("\n".join(d))
