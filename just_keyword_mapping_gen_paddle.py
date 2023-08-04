import json
import os.path

test_questions = open("test_questions_keyword.jsonl").readlines()
question_2020 = []
for test_question in test_questions:
    print(test_question)
    question = json.loads(test_question)
    # if "2020" in question:
    question_2020.append(question)

file_form = {}
first_label = json.load(open("form_list_keywords.json", "r"))
import json

DEVICE = "cuda"
DEVICE_ID = "0"
CUDA_DEVICE = f"{DEVICE}:{DEVICE_ID}" if DEVICE_ID else DEVICE





import re


def re_chinese(a_str):
    a_list = re.findall(r'[^\x00-\xff]', a_str)
    return "".join(a_list)


import paddle
from paddle.distributed import fleet

from paddlenlp.transformers import (
    ChatGLMConfig,
    ChatGLMForConditionalGeneration,
    ChatGLMTokenizer,
)


def batchfy_text(texts, batch_size):
    batch_texts = []

    batch_start = 0
    while batch_start < len(texts):
        batch_texts += [texts[batch_start: min(batch_start + batch_size, len(texts))]]
        batch_start += batch_size
    return batch_texts


class Predictor(object):
    def __init__(self, args=None):

        if args is None:
            self.tokenizer = ChatGLMTokenizer.from_pretrained("THUDM/chatglm-6b")
            self.batch_size = 1
            self.args = args
            self.src_length = 1024
            self.tgt_length = 512

            tensor_parallel_degree = paddle.distributed.get_world_size()
            tensor_parallel_rank = 0
            if tensor_parallel_degree > 1:
                strategy = fleet.DistributedStrategy()
                strategy.hybrid_configs = {
                    "dp_degree": 1,
                    "mp_degree": tensor_parallel_degree,
                    "pp_degree": 1,
                    "sharding_degree": 1,
                }
                fleet.init(is_collective=True, strategy=strategy)
                hcg = fleet.get_hybrid_communicate_group()
                tensor_parallel_rank = hcg.get_model_parallel_rank()

            config = ChatGLMConfig.from_pretrained("THUDM/chatglm-6b")
            dtype = config.dtype if config.dtype is not None else config.paddle_dtype

            self.model = ChatGLMForConditionalGeneration.from_pretrained(
                "THUDM/chatglm-6b",
                tensor_parallel_degree=tensor_parallel_degree,
                tensor_parallel_rank=tensor_parallel_rank,
                load_state_as_np=True,
                dtype=dtype,
            )

            self.model.eval()

    def preprocess(self, input_text):
        inputs = self.tokenizer(
            input_text,
            return_tensors="np",
            padding=True,
            max_length=self.src_length,
            truncation=True,
            truncation_side="left",
        )

        inputs_tensor = {}
        for key in inputs:
            inputs_tensor[key] = paddle.to_tensor(inputs[key])
        return inputs_tensor

    def infer(self, inputs):
        result = self.model.generate(
            **inputs,
            decode_strategy="sampling",
            top_k=1,
            max_length=self.tgt_length,
            bos_token_id=self.tokenizer.bos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
            use_cache=True,
        )

        result = result[0]
        return result

    def postprocess(self, infer_data):
        result = []

        for x in infer_data.tolist():
            res = self.tokenizer.decode(x, skip_special_tokens=True)
            res = res.strip("\n")
            result.append(res)
        out_dict = {"result": result}
        return out_dict

    def predict(self, texts):
        input_map = self.preprocess(texts)

        infer_result = self.infer(input_map)
        output = self.postprocess(infer_result)
        return output


# if __name__ == "__main__":
predictor = Predictor()

if __name__ == '__main__':
    result = []

    for question_one in question_2020:
        all_texts = []
        year_prefix = question_one['year']
        stock_name_list = question_one['stock_name']
        rows = []
        # jingzhunpipeibiaoge
        if "keyword" in question_one:
            print(question_one)
            for stock_name_one in stock_name_list:
                if 'match_annoy_name' in question_one:
                    if len(question_one['match_annoy_name']):
                        for stock_form in question_one['match_annoy_name']:
                            stock_form_key = os.path.join("./alltext/alltxt", stock_form)
                            for question_one_keyword in question_one['keyword']:
                                if stock_form_key in first_label:
                                    for one in first_label[stock_form_key]:
                                        if question_one_keyword in one:
                                            rows.append(one)
                if len(rows) == 0 and 'search_stock_name' in question_one:
                    stock_form_key = os.path.join("./alltext/alltxt", question_one['search_stock_name'])
                    for question_one_keyword in question_one['keyword']:
                        if stock_form_key in first_label:
                            for one in first_label[stock_form_key]:
                                if question_one_keyword in one:
                                    rows.append(one)
            message = "\n".join(rows)[:1024 + 512]
            all_texts = [
                "[Round 0]\n 根据以下几个表格:\n" + message + " 解决问题：\n" + question_one['question']
                + "    \n答：",
            ]
            # print(all_texts)

        if len(all_texts) == 0:
            all_texts = [
                "[Round 0]\n问：" + question_one['question'] + "    \n答：",
            ]
        outputs = predictor.predict(all_texts)

        print(all_texts[0])
        print(outputs)
        question_one['answer'] = outputs['result']
        result.append(question_one)
    d = []
    for i in result:
        d.append(json.dumps(i, ensure_ascii=False))
    print(d)
    open("search_form_text_gen_chinese_without_name.py.json", "w").write("\n".join(d))
