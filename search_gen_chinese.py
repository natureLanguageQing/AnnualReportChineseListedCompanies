from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.schema import Document
from langchain.vectorstores import FAISS

# 中文Wikipedia数据导入示例：
embedding_model_name = 'WangZeJun/simbert-base-chinese'
embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)

import json

test_questions = open("test_questions.jsonl").readlines()
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
first_label = json.load(open("first_label.json", "r"))
for stock_name_one in stock_name:
    for first_label_one in first_label:
        for first_label_key in first_label_one.keys():
            # print(first_label_key)
            file_form[first_label_key] = first_label_one[first_label_key]
            if stock_name_one in first_label_key:
                if stock_name_one in stock_mapping:
                    stock_mapping[stock_name_one].append(first_label_key)
                else:
                    stock_mapping[stock_name_one] = [first_label_key]

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
        for stock_name_one in stock_name:
            if stock_name_one in question_one['question']:
                if stock_name_one in stock_mapping:
                    # print(file_form[stock_mapping[stock_name_one][0]])
                    rows = []
                    for stock_form in stock_mapping[stock_name_one]:
                        rows.extend(file_form[stock_form])
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
                    vector_store = FAISS.from_documents(docs, embeddings)
                    query = question_one[
                        'question'].replace(stock_name_one, "")
                    file_form_one = vector_store.similarity_search(query)
                    all_texts = [
                        "[Round 0]\n 根据表格:" + chinese_row_mapping[file_form_one[0].page_content] + " 寻找问题：" + question_one[
                            'question'] + "    \n答：",
                    ]
        if len(all_texts) == 0:
            all_texts = [
                "[Round 0]\n问：" + question_one['question'] + "    \n答：",
            ]
        response, history = model.chat(tokenizer,
                                       all_texts[0],
                                       history=[],
                                       max_length=2048,
                                       top_p=0.7,
                                       temperature=0.95)
        print(all_texts[0])
        print(response)
        question_one['answer'] = response
d = []
for i in question_2020:
    d.append(json.dumps(i))
open("result.json", "w").write("\n".join(d))
