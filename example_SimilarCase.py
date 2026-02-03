from main_SimilarCase import update_casebank, retrieve_similar_case
import json

##############################################
# ./data/Input.json : new patients to to be diagnosed
# ./data/PatientInfo.json : historical patients 
# ./data/StructuredCaseCache.json : cached structured info of new patients in 'Input.json'
##############################################

with open('./data/PatientInfo.json', 'r') as f:
    loaded_data = json.load(f)
for ipid, case in loaded_data.items():
    update_casebank(
        cases=[(ipid, case)],
        raw_path="./data/CaseBankRaw.json",
        structured_path="./data/StructuredCaseBank.json"
    )
    break


with open('./data/Input.json', 'r') as f:
    loaded_data = json.load(f)
for ipid, record in loaded_data.items():
    similar_cases = retrieve_similar_case(
        new_case = (ipid, record),
        k = 3,
        casebank_path = "./data/StructuredCaseBank.json"
    )
    print("Similar Case ID:", similar_cases)

