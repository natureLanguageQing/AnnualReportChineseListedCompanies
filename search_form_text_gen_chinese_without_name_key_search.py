from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.schema import Document
from langchain.vectorstores import FAISS

# 中文Wikipedia数据导入示例：
embedding_model_name = 'WangZeJun/simbert-base-chinese'
embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)

import json

test_questions = open("test_questions_keyword.jsonl").readlines()
question_2020 = []
for test_question in test_questions:
    question = json.loads(test_question)
    # if "2020" in question:
    question_2020.append(question)

stock_mapping = json.load(open("stock_mapping.json", "r"))
stock_name = []
for stock_mapping_one in stock_mapping.values():
    if isinstance(stock_mapping_one, str):
        stock_name.append(stock_mapping_one)
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
text_list = json.load(open("text_list.json", "r"))
stock_text_mapping = {}
for i, j in text_list.items():
    for stock_name_one in stock_name:
        if stock_name_one in i:
            if stock_name_one not in stock_text_mapping:
                stock_text_mapping[stock_name_one] = [i]
            else:
                stock_text_mapping[stock_name_one].append(i)
    # paddle.device.cuda.empty_cache()
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
    model = AutoModel.from_pretrained("THUDM/chatglm2-6b", trust_remote_code=True).cuda()
    # 多显卡支持，使用下面三行代替上面两行，将num_gpus改为你实际的显卡数量
    # model_path = "THUDM/chatglm2-6b"
    # tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    # model = load_model_on_gpus(model_path, num_gpus=2)
    model.eval()
    for question_one in question_2020:
        all_texts = []
        year_prefix = []
        question_one = json.loads(question_one)
        if "2019年" in question_one['question']:
            year_prefix.append("2019年")
        if "2020年" in question_one['question']:
            year_prefix.append("2020年")
        if "2021年" in question_one['question']:
            year_prefix.append("2021年")
        # jingzhunpipeibiaoge
        if "keyword" in question_one:

            for stock_name_one in stock_name:
                if stock_name_one in question_one['question']:
                    if stock_name_one in stock_mapping:
                        rows = []
                        for stock_form in stock_mapping[stock_name_one]:
                            for year_prefix in year_prefix:
                                if year_prefix in stock_form:
                                    # print(question_one)
                                        for question_one_keyword in question_one['keyword']:
                                            for one in first_label[stock_form]:
                                                if question_one_keyword in one:
                                                    rows.append(one)

                            if len(rows) != 0:
                                rows = list(set(rows))
                                all_texts = [
                                    "[Round 0]\n 根据以下几个表格:" + "\n".join(rows) + " 解决问题：" +
                                    question_one[
                                        'question'] + "    \n答：",
                                ]
        # xiangliangpipeibiaoge
        if len(all_texts) == 0:
            for stock_name_one in stock_name:
                if stock_name_one in question_one['question']:
                    if stock_name_one in stock_mapping:
                        rows = []
                        for stock_form in stock_mapping[stock_name_one]:
                            for year_prefix in year_prefix:
                                if year_prefix in stock_form:
                                    rows.extend(first_label[stock_form])
                        idx = 0
                        rows = list(set(rows))
                        docs = []
                        chinese_row_mapping = {}
                        for row in rows:
                            metadata = {"source": f'doc_id_{idx}'}
                            idx += 1
                            if isinstance(row, str):
                                chinese_row = re_chinese(row)
                                chinese_row = chinese_row.replace(stock_name_one, "")
                                chinese_row_mapping[chinese_row] = row
                                docs.append(Document(page_content=chinese_row, metadata=metadata))
                        if len(docs):
                            vector_store = FAISS.from_documents(docs, embeddings)
                            query = question_one['question'].replace(stock_name_one, "")
                            file_form_one = vector_store.similarity_search(query)
                            chinese_row_prompt = "下一个表格".join([chinese_row_mapping[file_form_one.page_content] for file_form_one in file_form_one[:2]])
                            all_texts = [
                                "[Round 0]\n 根据以下几个表格:" + chinese_row_prompt[:1000] + " 解决问题：" +
                                question_one[
                                    'question'] + "    \n答：",
                            ]
        if len(all_texts) == 0:
            # gupiaomingzi
            for stock_name_one in stock_name:
                # wenti
                if stock_name_one in question_one['question']:
                    # gupiaoyingshewenjianming
                    if stock_name_one in stock_text_mapping:
                        # print(file_form[stock_mapping[stock_name_one][0]])
                        rows = []
                        for stock_form in stock_text_mapping[stock_name_one]:
                            for year_prefix in year_prefix:
                                if year_prefix in stock_form:
                                    rows.extend(text_list[stock_form])
                        idx = 0
                        rows = list(set(rows))
                        docs = []
                        chinese_row_mapping = {}
                        for row in rows:
                            metadata = {"source": f'doc_id_{idx}'}
                            idx += 1
                            if isinstance(row, str):
                                chinese_row = re_chinese(row)
                                chinese_row_mapping[chinese_row] = row
                                docs.append(Document(page_content=chinese_row, metadata=metadata))
                        if len(docs):
                            vector_store = FAISS.from_documents(docs, embeddings)
                            query = question_one[
                                'question'].replace(stock_name_one, "")
                            for year_prefix in year_prefix:
                                query = query.replace(year_prefix, "")
                            file_form_one = vector_store.similarity_search(query)
                            chinese_row_prompt = ''
                            for file_form_one in file_form_one:
                                chinese_row_prompt += chinese_row_mapping[file_form_one.page_content]
                            all_texts = [
                                "[Round 0]\n 根据:" + chinese_row_prompt[:1000] + " 寻找问题：" + question_one[
                                    'question'] + "    \n答：",
                            ]
        if len(all_texts) == 0:
            all_texts = [
                "[Round 0]\n问：" + question_one['question'] + "    \n答：",
            ]
        response, history = model.chat(tokenizer,
                                       all_texts[0],
                                       history=[],
                                       max_length=2048+1024,
                                       top_p=0.7,
                                       temperature=0.95)
        print(all_texts[0])
        print(response)
        question_one['answer'] = response
d = []
for i in question_2020:
    d.append(json.dumps(i,ensure_ascii=False))
open("search_form_text_gen_chinese_without_name.py.json", "w").write("\n".join(d))
