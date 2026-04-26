import numpy as np
import json
import pandas as pd
from graph_rag import kg_query
import os
from HyperGraph import HyperGraphRAG
from _utils import convert_response_to_json
from _llm import deepseek_response, doubao_response, qwen_response, siliconflow_response_if_catch, deepseek_response_if_catch, siliconflow_response, local_response, local_response_if_catch
from prompt import prompt
from datetime import datetime

####################################### STEPS #######################################
 
def generate_cnode(patient_info: tuple[str, str]):
    inquiry_prompt = prompt['inquiry_user']
    full_prompt = inquiry_prompt.format(case_id = patient_info[0], case_detail=patient_info[1], similar_case = None)
    return qwen_response(user_prompt = full_prompt, system_prompt = prompt["generate_cnode_binary"])

def calculate_entropy(column):
    value_counts = column.value_counts(normalize=True)
    entropy = -np.sum(value_counts * np.log2(value_counts + 1e-10))  # 加上小常数以避免log(0)
    return entropy

def process_binary_cnode(df):
    def discretize_age(age_series):
        bins = [0, 18, 60, 100]  # 定义年龄区间
        labels = [i / 2 for i in range(len(bins) - 1)]
        return pd.cut(age_series, bins=bins, labels=labels, right=False)
    

    df['年龄'] = df['年龄'].astype(int)
    if '年龄' in df.columns:
        df['年龄'] = discretize_age(df['年龄'])
    #df['年龄'] = (df['年龄'] - df['年龄'].min()) / (df['年龄'].max() - df['年龄'].min())
    df['起病形式'] = df['起病形式'].map({'急性起病': 1, '慢性起病': 0, '亚急性起病': 0.5, "null": None})
    df['病前性格'] = df['病前性格'].map({'内向': 1, '外向': 0, "null": None})
    df['性别'] = df['性别'].map({'男': 1, '女': 0})
    df['病程'] = df['病程'].map({'间断病程': 1, '持续病程': 0, "null": None})
    df[['过敏史', '合并躯体病', '发病诱因', '躯体病家族史', '精神疾病家族史', "心境低落", "兴趣或愉快感丧失", "精力不足或易疲劳",
        "自信心不足或自责", "自杀意念或行为", "注意力下降或犹豫不决", "精神运动性激越或迟滞", "睡眠障碍", "食欲或体重改变", "性欲减退", "注意力分散或随境转移",
        "夸大观念或自我评价膨胀", "鲁莽行为", "易激惹", "攻击性强", "睡眠需求显著减少",
        "言语迫促",
        "心境高涨",
        "思维奔逸或联想加快",
        "活动增多或精力旺盛"]] = df[['过敏史', '合并躯体病', '发病诱因', '躯体病家族史', '精神疾病家族史', "心境低落",
        "兴趣或愉快感丧失",
        "精力不足或易疲劳",
        "自信心不足或自责",
        "自杀意念或行为",
        "注意力下降或犹豫不决",
        "精神运动性激越或迟滞",
        "睡眠障碍",
        "食欲或体重改变",
        "性欲减退",
        "注意力分散或随境转移",
        "夸大观念或自我评价膨胀",
        "鲁莽行为",
        "易激惹",
        "攻击性强",
        "睡眠需求显著减少",
        "言语迫促",
        "心境高涨",
        "思维奔逸或联想加快",
        "活动增多或精力旺盛"]].replace({"true": 1, "false": 0, "null": None}).infer_objects(copy=False)
    return df

def calculate_similarity(row1, row2, entropy_weights):
    similarity = 0
    for feature in row1.index:
        if pd.isna(row1[feature]) or pd.isna(row2[feature]) or (feature == 'IPID'):
            continue 
        if isinstance(row1[feature], (int, float)):
            similarity += (1 - abs(row1[feature] - row2[feature])) * entropy_weights[feature]
        else:
            similarity += (1 if row1[feature] == row2[feature] else 0) * entropy_weights[feature]
    return similarity

def retrieve_similar_case(dataname, ipid, k = 5):
    with open("data/datacase.json", "r", encoding="utf-8") as f:
        case_set = json.load(f)
    df = pd.DataFrame(case_set)
    df = process_binary_cnode(df)
    entropy_results = {column: calculate_entropy(df[column]) for column in df.columns if column != 'IPID'}'
    with open(f'data/Structured{dataname}.json','r', encoding = "utf-8") as f:
        structured_case = json.load(f)
    new_case = structured_case[ipid]
    new_case_df = process_binary_cnode(pd.DataFrame([new_case]))
    similarities = []
    for i in range(len(df)):
        similarity = calculate_similarity(new_case_df.iloc[0], df.iloc[i], entropy_results)
        similarities.append((df.iloc[i]['IPID'], similarity))
    similarities.sort(key=lambda x: x[1], reverse=True)  
    top_k_similar_cases = [case[0] for case in similarities if case[0] != ipid][:k]
    return top_k_similar_cases

def extract_experience(top_k_index, tag : str = 'T'):
    if tag == 'D':
        with open('data/experience/diagnose_experience.json', 'r', encoding='utf-8') as json_file:
            experience_dict = json.load(json_file)
    if tag == 'T':
        with open('data/experience/treatment_experience.json', 'r', encoding='utf-8') as json_file:
            experience_dict = json.load(json_file)
    dict_list = []
    for ipid_index in top_k_index:
        if ipid_index in experience_dict.keys():
            dict_list.extend(experience_dict[ipid_index]) 
        else:
            print(ipid_index,'GOT NO EXPERIENCE!')
    
    return dict_list


############### work flow ###############
class AdaptiveDiagnosisSystem:
    def __init__(self, hypergraph_rag, symptom_list, ipid, confidence_threshold=0.95):
        self.hypergraph = hypergraph_rag
        self.symptom_pool = symptom_list  
        self.confidence_threshold = confidence_threshold
        self.diagnosis_flow = []
        self.dialog_history = []
        self.ipid = ipid
    def generate_diagnosis_prompt(self, case_data: str) -> dict:
        user_prompt = f'''
            当前病例信息：
            {case_data}
            
            你是一个知识渊博的精神疾病专家。
            你的任务是：
            1. 阅读提供的病例资料，包括患者的基本信息、病史、身心检查记录、首次病程等信息，并根据这些记录对患者当前的发病诊断做出推理和判断。
            请在进行诊断结果的同时给出你对于这一诊断的置信度，若证据不充分或者你对自己的结果不够确认，你可以选择检索外部知识源来辅助诊断。
            在没有检索任何外部资料就进行诊断时，尽可能在置信度的评估时采取保守谨慎的态度

            2. 检查你在本次推断中第1步中给出的置信度，若置信度>{self.confidence_threshold}，才可以选择直接输出该病例的诊断结果，此时应选择本次输出中的decision字段为 'final_ans' 选项
            注意一定要置信度>{self.confidence_threshold}才能选择直接输出final_ans选项！请反复确认
            
            3. 检查你在第1步中给出的置信度，若置信度>{self.confidence_threshold}，才可以选择直接输出该病例的诊断结果，此时应选择本次输出中的decision字段为 'final_ans' 选项

            若目前结果的置信度<={self.confidence_threshold}，你可以选择以下两种信息源进行相关知识的检索

            - 临床经验知识： 我根据历史病例库里的症状相似病例的病程记录提取出了一些经验知识，它们被构建为图谱，你可以在{', '.join(self.symptom_pool)}这个症状和诊断的节点列表里选择一些相关的节点进行查询，可得到这些症状节点对应的诊断推导相关的经验知识和诊断节点对应的辨别区分相近诊断之间相关的知识，此时需要选择本次输出中的decision字段为 'experience' 选项
            
            - 指南定义知识： 我根据ICD10临床指南对各种心境障碍疾病的定义构建了知识图谱，你可以生成一个相关的问题，（如“[F31.0, 双相情感障碍,目前为轻躁狂发作]的诊断标准是什么”，或“怎么区分[F31.302, 双相情感障碍,目前为轻度抑郁发作]和[F31.303, 双相情感障碍,目前为不伴有躯体症状的轻度抑郁发作]，它们之间有什么辨别差异？”），此时需要选择本次输出中的decision字段为 'guidelines' 选项
            
            notice: 请尽量确保诊断的推断过程都是有相似病例的临床经验知识作为依据的。

            4. 请按以下JSON格式响应：
            {{
                "decision": ["final_ans"/"experience"/"guidelines"],
                "selected_symptoms": ["相关症状1", "相关症状2"],  // 仅选择experience时需要
                "guidance_question": "需要提问知识图谱的问题内容",  // 仅选择guidelines时需要
                "diagnosis_result": {{
                    "ICD": "代码",
                    "diagnosis": "诊断名称",
                   "confidence": 0.0-1.0
                }},
                "reasoning": "当前决策的逻辑链（使用'替代引号）"
            }}
         '''
        return {
            "system": prompt["diagnose_system"],
            "user": user_prompt,
            "full_history": self.dialog_history.copy()
        }

    def process_case(self, case_data: str) -> dict:
        
        current_prompt = self.generate_diagnosis_prompt(case_data)
        response = self._call_llm_with_history(current_prompt)
        while True:
            # 记录对话历史
            if (len(self.dialog_history) >= 11):
                return self._format_final_result(response), self.dialog_history
            decision = response.get("decision")
            print(decision)
            if decision == "final_ans":
                if float(response["diagnosis_result"]["confidence"]) >= self.confidence_threshold:
                    return self._format_final_result(response), self.dialog_history
                else:
                    if (float(response["diagnosis_result"]["confidence"]) < self.confidence_threshold)and(decision == "final_ans"):
                        print(float(response["diagnosis_result"]["confidence"]))
                        return self.process_case(case_data)
            elif decision == "experience":
                knowledge = self.hypergraph.get_related_knowledge(response["selected_symptoms"])
                knowledge_prompt = f"检索到的历史病例记录中相关的经验知识有:{knowledge}"
                response = self._next_round_with_knowledge(knowledge_prompt)
                
            elif decision == "guidelines":
                knowledge = self._query_guidance(response["guidance_question"] + "请尽量使用中文回答！")
                """# 更新对话历史
                self._update_dialog_history(
                    user_input=f"询问指南：{response['guidance_question']}",
                    ai_response=answer
                )"""
                knowledge_prompt = f"查询知识图谱后得到的回答是:{knowledge}"
                response = self._next_round_with_knowledge(knowledge_prompt)
            else:
                print(decision)
                raise ValueError("Invalid decision value in response")

    def _retrieve_knowledge(self, symptoms: list) -> dict:
        valid_symptoms = [s for s in symptoms if s in self.symptom_pool]
        return {
            "type": "symptom_knowledge",
            "content": self.hypergraph.get_related_knowledge(valid_symptoms),
            "source": "experience_graph"
        }

    def _query_guidance(self, question: str) -> dict:
        return kg_query(question, WORKING_DIR='./psychrag_kg')

    def _next_round_with_knowledge(self, knowledge) -> dict:
        new_prompt = f'''
            当前检索到的经验或知识为：
            {knowledge}

            你是一个知识渊博的精神疾病专家。
            1. 阅读历史消息记录中提供的病例资料，包括患者的基本信息、病史、身心检查记录、首次病程等信息和推断的中间过程，并根据这些记录对患者当前的发病诊断做出推理和判断。
            请在进行诊断结果的同时给出你对于这一诊断的置信度，若证据不充分或者你对自己的结果不够确认，你可以选择检索外部知识源来辅助诊断

            2. 检查你在本次推断中第1步中给出的置信度，若置信度>{self.confidence_threshold}，才可以选择直接输出该病例的诊断结果，此时应选择本次输出中的decision字段为 'final_ans' 选项
            注意一定要置信度>{self.confidence_threshold}才能选择直接输出final_ans选项！请反复确认
            
            3. 检查你在第1步中给出的置信度，若置信度>{self.confidence_threshold}，才可以选择直接输出该病例的诊断结果，此时应选择本次输出中的decision字段为 'final_ans' 选项

            若目前结果的置信度<={self.confidence_threshold}，你可以选择以下两种信息源进行相关知识的检索

            - 临床经验知识： 我根据历史病例库里的症状相似病例的病程记录提取出了一些经验知识，它们被构建为图谱，你可以在{', '.join(self.symptom_pool)}这个症状和诊断的节点列表里选择一些相关的节点进行查询，可得到这些症状节点对应的诊断推导相关的经验知识和诊断节点对应的辨别区分相近诊断之间相关的知识，此时需要选择本次输出中的decision字段为 'experience' 选项
            
            - 指南定义知识： 我根据ICD10临床指南对各种心境障碍疾病的定义构建了知识图谱，你可以生成一个相关的问题，（如“[F31.0, 双相情感障碍,目前为轻躁狂发作]的诊断标准是什么”，或“怎么区分[F31.302, 双相情感障碍,目前为轻度抑郁发作]和[F31.303, 双相情感障碍,目前为不伴有躯体症状的轻度抑郁发作]，它们之间有什么辨别差异？”），此时需要选择本次输出中的decision字段为 'guidelines' 选项
            
            notice: 请尽量确保诊断的推断过程都是有相似病例的临床经验知识作为依据的.
            
            4. 请按以下JSON格式响应：
            {{
                "decision": ["final_ans"/"experience"/"guidelines"],
                "selected_symptoms": ["相关症状1", "相关症状2"],  // 仅选择experience时需要
                "guidance_question": "需要提问知识图谱的问题内容",  // 仅选择guidelines时需要
                "diagnosis_result": {{
                    "ICD": "代码",
                    "diagnosis": "诊断名称",
                   "confidence": 0.0-1.0
                }},
                "reasoning": "当前决策的逻辑链（使用'替代引号）"
            }}
            notice: 请在输出reasoning内容时详细叙述你是如何结合检索到的经验知识或者指南知识，并根据自己的推理一步步得到结果的
         '''
        return self._call_llm_with_history({
            "system": prompt["diagnose_system"],
            "user": new_prompt,
            "full_history": self.dialog_history.copy()
        })

    def _format_final_result(self, response: dict) -> dict:
        """格式化最终输出"""
        return {
            "IPID": self.ipid,
            "diagnose": [
                response["diagnosis_result"]["ICD"],
                response["diagnosis_result"]["diagnosis"]
            ],
            "confidence": response["diagnosis_result"]["confidence"],
            "diagnosis_path": self.diagnosis_flow,
            "explanations": response["reasoning"]
        }
        
    def _call_llm_with_history(self, prompt: dict) -> dict:
        raw_response = local_response(
            user_prompt=prompt["user"],
            system_prompt=prompt["system"],
            history_messages=prompt["full_history"]
        )
        result_in_json = convert_response_to_json(raw_response)
        self._update_dialog_history(
            user_input=prompt["user"],
            ai_response=result_in_json
        )
        return result_in_json

    def _update_dialog_history(self, user_input: str, ai_response: dict):
        self.dialog_history.append({
            "role": "user",
            "content": user_input
        })
        self.dialog_history.append({
            "role": "assistant",
            "content": json.dumps(ai_response, ensure_ascii=False)
        })

class AdaptiveTreatmentSystem:
    def __init__(self, hypergraph_rag, node_list, ipid, model, confidence_threshold=0.90):
        self.hypergraph = hypergraph_rag
        self.node_pool = node_list  # 预加载的症状节点列表
        self.confidence_threshold = confidence_threshold
        self.drug_set = prompt['drug_set']
        self.dialog_history = []
        self.ipid = ipid
        self.model = model
    def generate_diagnosis_prompt(self, case_data: str) -> dict:
        user_prompt = f'''
            当前病例信息：
            {case_data}
            
            你是一个知识渊博的精神疾病专家。
            你的任务是：
            1. 阅读提供的病例资料，包括患者的基本信息、病史、身心检查记录、首次病程等信息，并根据这些记录进行推断，为患者本次入院推荐最合适、最有效的系统用药方案，包含对应的五个药物的主成分名称。
            请在进行用药建议的同时给出你对于这一用药方案的置信度（0.0-1.0），若证据不充分或者你对自己的结果不够确认，你可以在输出当前的药物推荐方案和置信度后，检索外部知识源来辅助诊断。
            注意：考虑到医院的药品种类有限，所有推荐的药物都需要从给出的药物列表中选择。

            2. 检查你在本次生成用药方案中第1步中给出的置信度，若置信度>{self.confidence_threshold}，才可以选择直接输出该病例的用药推荐方案，此时应选择本次输出中的decision字段为 'final_ans' 选项
            注意一定要置信度>{self.confidence_threshold}才能选择直接输出final_ans选项！
            
            3. 若目前结果的置信度<={self.confidence_threshold}，你可以选择以下两种信息源进行相关知识的检索和参考
            - 临床经验知识： 我根据历史病例库里的心境障碍类别下症状相似病例的病程记录提取出了一些经验知识，它们被构建为图谱，你可以选择一些相关的节点进行查询，此时需要选择本次输出中的decision字段为 'experience' 选项
                可选择的知识节点有：
                * 医院药物库里的具体药品：[{self.drug_set}]
                * 相关的症状节点：[{', '.join(self.node_pool)}]
                给出一个包含所有需要查询的药物和症状节点的列表，可得到从相似的历史病例记录中提取出的临床经验知识图谱内与这些节点相关的全部历史患者用药经验
            - 指南定义知识： 我根据精神心理医学教材里心境障碍相关的知识构建了知识图谱，它包含一些药物的说明和临床使用药物的指导。你可以生成一个相关的问题，（如“有哪些类别的药物适合联合治疗精神分裂症？”，或“齐拉西酮在使用时有哪些注意事项和药物禁忌？”等），此时需要选择本次输出中的decision字段为 'guidelines' 选项
            
            notice: 请不要虚构任何内容，并尽量进行两种知识源的交叉验证。确保每一步替换药物都拥有依据。
            由于临床经验知识图谱是静态的，请尽量避免重复查询相同节点获取重复的知识。
            基于患者用药的连续性，在推荐用药时请优先考虑患者当前正在使用的药物，除非有强烈证据表明这些药物无效或不安全。            
            在每一次生成答案时，同时更新“useful_experience”列表，把截至目前推断过程中你认为之前检索到的有用的经验知识的原句依次列在这里，若没有则空着。
            所有提取临床经验的历史病例均属于心境障碍诊断，若你觉得当前病例不符合主要诊断为心境障碍类别，可选择性参考。

            4. 请按以下JSON格式响应：
            {{
                "decision": ["final_ans"/"experience"/"guidelines"],
                "selected_nodes": ["相关症状1", "相关症状2", "相关药物1"],  // 仅选择experience时需要
                "guidance_question": "需要提问知识图谱的问题内容，请使用中文提问",  // 仅选择guidelines时需要
                "treatment_result": {{   //当前的药物推荐方案
                    "drug_list": ["药物主成分1的中文名称", "药物主成分2的中文名称",...]
                    "confidence": 0.0-1.0
                }}
                "reasoning": "当前用药推荐方案的逻辑链",
                "useful_experience": [""experience1", "experience2", ...]
            }}
         '''
        return {
            "system": prompt["treatment_system"],
            "user": user_prompt,
            "full_history": self.dialog_history.copy()
        }

    def process_case(self, case_data: str) -> dict:
        current_prompt = self.generate_diagnosis_prompt(case_data)
        response = self._call_llm_with_history(current_prompt)
        while True:
            decision = response.get("decision")
            print(decision)
            if decision == "final_ans":
                if float(response["treatment_result"]["confidence"]) >= self.confidence_threshold:
                    return self._format_final_result(response), self.dialog_history
                else:
                    if (float(response["treatment_result"]["confidence"]) < self.confidence_threshold)and(decision == "final_ans"):
                        print(float(response["treatment_result"]["confidence"]))
                        return self.process_case(case_data)
            elif decision == "experience":
                knowledge = self.hypergraph.get_related_knowledge(response["selected_nodes"])
                knowledge_prompt = f"检索到的历史病例记录中相关的经验知识:{knowledge}"
                response = self._next_round_with_knowledge(knowledge_prompt)
                
            elif decision == "guidelines":
                knowledge = self._query_guidance(response["guidance_question"] + "请尽量使用中文回答！")
                knowledge_prompt = f"查询知识图谱后得到的回答是:{knowledge}"
                response = self._next_round_with_knowledge(knowledge_prompt)
            else:
                print(decision)
                raise ValueError("Invalid decision value in response")
            if (len(self.dialog_history) >= 7):
                return self._format_final_result(response), self.dialog_history

    def _query_guidance(self, question: str) -> dict:
        response = kg_query(question, WORKING_DIR='./psychrag_kg')
        return response

    def _next_round_with_knowledge(self, knowledge) -> dict:
        new_prompt = f'''
            当前检索到的经验或知识为：
            {knowledge}
            你是一个知识渊博的精神疾病专家。
            你的任务是：
            1. 阅读提供的病例资料，包括患者的基本信息、病史、身心检查记录、首次病程等信息，并根据这些记录进行推断，为患者本次入院推荐最合适、最有效的用药方案，包含对应的五个药物的主成分名称。
            请在进行用药建议的同时给出你对于这一用药方案的置信度（0.0-1.0），若证据不充分或者你对自己的结果不够确认，你可以在输出当前的药物推荐方案和置信度后，选择检索外部知识源来辅助诊断。
            注意：考虑到医院的药品种类有限，所有推荐的药物都需要从给出的药物列表中选择。
            
            2. 检查你在本次生成用药方案中第1步中给出的置信度，若置信度>{self.confidence_threshold}，才可以选择直接输出该病例的用药推荐方案，此时应选择本次输出中的decision字段为 'final_ans' 选项
            注意一定要置信度>{self.confidence_threshold}才能选择直接输出final_ans选项！请反复确认！
            
            3. 若目前结果的置信度<={self.confidence_threshold}，你可以选择以下两种信息源进行相关知识的检索
            - 临床经验知识： 我根据历史病例库里的心境障碍类别下症状相似病例的病程记录提取出了一些经验知识，它们被构建为图谱，你可以选择一些相关的节点进行查询，此时需要选择本次输出中的decision字段为 'experience' 选项
                可选择的知识节点有：
                * 医院药物库里的具体药品：[{self.drug_set}]
                * 相关的症状节点：[{', '.join(self.node_pool)}]
                给出一个包含所有需要查询的药物和症状节点的列表，可得到从相似的历史病例记录中提取出的临床经验知识图谱内与这些节点相关的全部历史患者用药经验
            - 指南定义知识： 我根据精神心理医学教材里心境障碍相关的知识构建了知识图谱，它包含一些药物的说明和临床使用药物的指导。你可以生成一个相关的问题，（如“有哪些药物适合治疗精神分裂症？”，或“齐拉西酮在使用时有哪些注意事项？”等），此时需要选择本次输出中的decision字段为 'guidelines' 选项
            
            notice: 请不要虚构任何内容，并尽量进行两种知识源的交叉验证。确保最终的药物推荐方案经过多源知识的验证，且每一步替换药物都拥有依据。
            请注意临床经验知识图谱是静态的，重复查询相同节点无法获得新的经验知识，请避免重复查询相同节点。
            基于患者用药的连续性，在推荐用药时，请优先考虑患者当前正在使用的药物，除非有强烈证据表明这些药物无效或不安全。
            在每一次生成答案时，同时更新“useful_experience”列表，把截至目前推断过程中你认为之前检索到的有用的经验知识的原句依次列在这里，若没有则空着。
            所有提取临床经验的历史病例均属于心境障碍诊断，若你觉得当前病例不符合主要诊断为心境障碍类别，可选择性参考。
            
            4. 请按以下JSON格式响应：
            {{
                "decision": ["final_ans"/"experience"/"guidelines"],
                "selected_nodes": ["相关症状1", "相关症状2", "相关药物1"],  // 仅选择experience时需要
                "guidance_question": "需要提问知识图谱的问题内容，请使用中文提问",  // 仅选择guidelines时需要
                "treatment_result": {{   //当前的药物推荐方案
                    "drug_list": ["药物主成分1的中文名称", "药物主成分2的中文名称",...]
                    "confidence": 0.0-1.0
                }}
                "reasoning": "当前用药推荐方案的逻辑链",
                "useful_experience": [""experience1", "experience2", ...]
            }}
            notice: 请在输出reasoning内容时详细叙述你是如何结合检索到的经验知识或者指南知识，并根据自己的推理一步步得到结果的
         '''
        return self._call_llm_with_history({
            "system": prompt["treatment_system"],
            "user": new_prompt,
            "full_history": self.dialog_history.copy()
        })
    def _format_final_result(self, response: dict) -> dict:
        return {
            "IPID": self.ipid,
            "drugname": response["treatment_result"]["drug_list"],
            "confidence": response["treatment_result"]["confidence"],
            "explanations": response["reasoning"],
            "useful_experience": response["useful_experience"]
        }
        
    def _call_llm_with_history(self, prompt: dict) -> dict:
        raw_response = local_response(
            model_name=self.model,
            user_prompt=prompt["user"],
            system_prompt=prompt["system"],
            history_messages=prompt["full_history"]
        )
        print('raw_response:',raw_response)
        result_in_json = convert_response_to_json(raw_response)
        print('result in json:', result_in_json)
        self._update_dialog_history(
            user_input=prompt["user"],
            ai_response=result_in_json
        )
        return result_in_json

    def _update_dialog_history(self, user_input: str, ai_response: dict):
        self.dialog_history.append({
            "role": "user",
            "content": user_input
        })
        
        self.dialog_history.append({
            "role": "assistant",
            "content": json.dumps(ai_response, ensure_ascii=False)
        })