from transformers import AutoTokenizer, AutoModel
import json
tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm2-6b", trust_remote_code=True)
model = AutoModel.from_pretrained("THUDM/chatglm2-6b", trust_remote_code=True, device='cuda')
model.eval()
test_questions = open("test_questions.jsonl").readlines()
question_2020 = []
for test_question in test_questions:
    test_question = json.loads(test_question)
    question_2020.append(test_question)
    all_texts = [
        "[当前是一个抽取任务]\n抽取 \t"+test_question['question']+"\t 中的公司名称，如果没有回答无。",
    ]
    response, history = model.chat(tokenizer,
                                   all_texts[0],
                                   history=[],
                                   max_length=512,
                                   top_p=0.7,
                                   temperature=0.95)
    print(response)