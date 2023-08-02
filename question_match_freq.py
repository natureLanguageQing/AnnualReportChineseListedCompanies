import pandas as pd
import json

most_common = pd.read_csv('most_common.csv')
keyword_list = []
d = []
test_questions = open("test_questions.jsonl").readlines()
for test_question in test_questions:
    test_question = json.loads(test_question)
    test_question["keyword"] = []
    for most_common_one in most_common.values.tolist()[:2000]:
        if isinstance(most_common_one[1], str):
            if len(most_common_one[1]) > 2:
                if most_common_one[1] in test_question['question']:
                    print(most_common_one)
                    test_question["keyword"].append(most_common_one[1])
                    keyword_list.append(most_common_one[1])
    d.append(test_question)
    print(test_question)
test_questions_keyword = open("test_questions_keyword.jsonl", "w")
for test_question in d:
    test_questions_keyword.write(json.dumps(test_question, ensure_ascii=False) + "\n")
keyword_list = list(set(keyword_list))
json.dump(keyword_list, open("keyword_list.json", "w"), ensure_ascii=False)
