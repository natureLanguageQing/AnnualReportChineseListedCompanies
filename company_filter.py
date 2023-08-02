import json

test_questions = open("test_questions.jsonl").readlines()
question_2020 = []
for test_question in test_questions:
    question = json.loads(test_question)
    question_2020.append(question)

stock_mapping = json.load(open("stock_mapping.json", "r"))
stock_name = {}
stock_in_question = []
for stock_mapping_one in stock_mapping.values():
    if isinstance(stock_mapping_one, str):
        stock_name[stock_mapping_one] = []

        for question_one in question_2020:
            if stock_mapping_one in question_one['question']:
                stock_in_question.append(stock_mapping_one)
                if "2019" in question_one['question']:
                    stock_name[stock_mapping_one].append("2019")
                if "2020" in question_one['question']:
                    stock_name[stock_mapping_one].append("2020")
                if "2021" in question_one['question']:
                    stock_name[stock_mapping_one].append("2021")
# stock_name = list(set(stock_name))
online_stock_name = {}
for i , j in stock_name.items():
    if j:
        online_stock_name[i] = list(set(j))
stock_in_question = list(set(stock_in_question))
json.dump(online_stock_name, open("in_stock_name.json", "w"), ensure_ascii=False,indent=2)
json.dump(stock_in_question, open("stock_in_question.json", "w"), ensure_ascii=False,indent=2)
