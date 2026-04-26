from main_PsychRAG import retrieve_similar_case, AdaptiveTreatmentSystem, extract_experience
import os
from _utils import convert_response_to_json
import time
from HyperGraph import HyperGraphRAG
import json
from _utils import convert_response_to_json
top_k_dict = {}
import nest_asyncio
nest_asyncio.apply()
print('begiin!')
cnt = 0
backbone = 'Qwen/Qwen3-32B'
result_dict = {}
def read_jsonl(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            data.append(json.loads(line))
    return data
loaded_data = read_jsonl(f'data/testdata.jsonl')
print(len(loaded_data))
standard_answers_file = 'data/groundtruth.json' 
with open(standard_answers_file, 'r', encoding='utf-8') as f:
    standard_answers = json.load(f)
diagnose_dict = {}
for case in standard_answers:
    ipid = case['IPID']
    std_ans = case['primary_icd']
    diagnose_dict[ipid] = [std_ans, case['primary_diagnosis']]
result_dict = {}
case_file_path = "data/datacase.json"
knowledge_dir = "./psychrag_kg"
for item in loaded_data:
    ipid = item['conversations'][0]['value']
    result = {}
    result['ipid'] = item['conversations'][0]['value']
    result['record'] = item['conversations'][1]['value']
    result['groundtruth'] = item['conversations'][2]['value'].split(", ")
    result['diagnose'] = diagnose_dict[ipid][0:2]
    top_k_index = retrieve_similar_case(dataname=dataname, ipid = ipid, k = 5)
    dict = extract_experience(top_k_index)
    hg = HyperGraphRAG()
    for treatment in dict:
        hg.insert_context(treatment)
    symptom_nodes = hg.get_all_nodes(node_type='symptom')
    rag_system = AdaptiveTreatmentSystem(hg, symptom_nodes, ipid, model = backbone)
    result['test_result'], result['test_dialog'] = rag_system.process_case(result['record'])
    result_dict[ipid] = result
    cnt = cnt +1
with open(f'PsychRAG_{dataname}_{backbone}_{date}.json', "w", encoding="utf-8") as json_file:
    json.dump(result_dict, json_file, ensure_ascii=False, indent=4)

