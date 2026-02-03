import numpy as np
import json
import pandas as pd
pd.set_option('future.no_silent_downcasting', True)
import os
from _utils import convert_response_to_json, load_json, write_json
from _llm import deepseek_response, doubao_response, qwen_response
from typing import List, Dict, Tuple, Callable
from prompt import prompt

def structure_case(patient_info: Tuple[str, str], response_func: Callable = qwen_response) -> Dict:
    print(f"structuring {patient_info[0]}...")
    inquiry_prompt = prompt['inquiry_user']
    full_prompt = inquiry_prompt.format(
        case_id=patient_info[0],
        case_detail=patient_info[1],
        similar_case=None
    )
    return response_func(user_prompt=full_prompt, system_prompt=prompt["structure_case"])

def calculate_entropy(column: pd.Series) -> float:
    value_counts = column.value_counts(normalize=True)
    return -np.sum(value_counts * np.log2(value_counts + 1e-10))

def process_binary_cnode(df: pd.DataFrame) -> pd.DataFrame:

    def discretize_age(age_series: pd.Series) -> pd.Series:
        bins = [0, 18, 60, 100]
        labels = [i / 2 for i in range(len(bins) - 1)]
        return pd.cut(age_series, bins=bins, labels=labels, right=False)

    if '年龄' in df.columns:
        df['年龄'] = df['年龄'].astype(int)
        df['年龄'] = discretize_age(df['年龄'])

    mapping_features = {
        '起病形式': {'急性起病': 1, '慢性起病': 0, '亚急性起病': 0.5, "null": None},
        '病前性格': {'内向': 1, '外向': 0, "null": None},
        '性别': {'男': 1, '女': 0},
        '病程': {'间断病程': 1, '持续病程': 0, "null": None}
    }
    for col, mapping in mapping_features.items():
        if col in df.columns:
            df[col] = df[col].map(mapping)

    bool_columns = [
        '过敏史', '合并躯体病', '发病诱因', '躯体病家族史', '精神疾病家族史',
        "心境低落", "兴趣或愉快感丧失", "精力不足或易疲劳",
        "自信心不足或自责", "自杀意念或行为", "注意力下降或犹豫不决",
        "精神运动性激越或迟滞", "睡眠障碍", "食欲或体重改变", "性欲减退",
        "注意力分散或随境转移", "夸大观念或自我评价膨胀", "鲁莽行为", "易激惹",
        "攻击性强", "睡眠需求显著减少", "言语迫促", "心境高涨",
        "思维奔逸或联想加快", "活动增多或精力旺盛"
    ]

    existing_bool_cols = [col for col in bool_columns if col in df.columns]
    df[existing_bool_cols] = df[existing_bool_cols].replace({"true": 1, "false": 0, "null": None}).infer_objects(copy=False)
    return df

def calculate_similarity(row1: pd.Series, row2: pd.Series, entropy_weights: Dict[str, float]) -> float:
    similarity = 0.0
    
    for feature in row1.index:
        if feature == 'IPID' or pd.isna(row1.loc[feature]) or pd.isna(row2.loc[feature]):
            continue
        weight = entropy_weights.get(feature, 1.0)
        if isinstance(row1.loc[feature], (int, float, np.number)) and isinstance(row2.loc[feature], (int, float, np.number)) :
            similarity += (1 - abs(row1.loc[feature] - row2.loc[feature])) * weight
        else:
            similarity += (1 if row1.loc[feature] == row2.loc[feature] else 0) * weight

    return similarity


def retrieve_similar_case(new_case: Tuple[str, str], k: int, casebank_path: str, cache_path = './data/StructuredCaseCache.json') -> List[str]:
    if not os.path.exists(casebank_path):
        raise FileNotFoundError(f"Case Bank Not Found: {casebank_path}")

    with open(casebank_path, "r", encoding="utf-8") as f:
        case_bank = json.load(f)

    case_bank_df = process_binary_cnode(pd.DataFrame(case_bank))
    entropy_results = {
        column: calculate_entropy(case_bank_df[column])
        for column in case_bank_df.columns if column != 'IPID'
    }

    with open(cache_path, "r", encoding="utf-8") as f:
        case_cache = json.load(f)
        if new_case[0] in case_cache:
            print(f"case {new_case[0]} in cache")
            structured_case = case_cache[new_case[0]]
        else:
            
            structured_case = convert_response_to_json(structure_case(new_case))
            structured_case["IPID"] = new_case[0]
            case_cache[new_case[0]] = structured_case
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(case_cache, f, ensure_ascii=False, indent=2)
            print(f"cache new case: {new_case[0]} .")
    new_case_df = process_binary_cnode(pd.DataFrame([structured_case]))
        
    similarities = [
        (row['IPID'], calculate_similarity(new_case_df.iloc[0], row, entropy_results))
        for _, row in case_bank_df.iterrows()
    ]
    similarities.sort(key=lambda x: x[1], reverse=True)

    return [ipid for ipid, _ in similarities[:k]]

def update_casebank(
    cases: List[Tuple[str, str]],
    raw_path: str = './data/CaseBankRaw.json',
    structured_path: str = './data/StructuredCaseBank.json'
) -> None:

    os.makedirs(os.path.dirname(raw_path), exist_ok=True)
    os.makedirs(os.path.dirname(structured_path), exist_ok=True)

    raw_cases = load_json(raw_path)
    structured_cases = load_json(structured_path)
    existing_ipids = {case["IPID"] for case in raw_cases}

    new_raw_cases = []
    new_structured_cases = []
    for case in cases:
        if case[0] in existing_ipids:
            print(f"case {case[0]} exists, skip!")
            continue
        new_raw_cases.append({"IPID": case[0], "raw_text": case[1]})
        structured_case = convert_response_to_json(structure_case((case[0], case[1])))
        structured_case["IPID"] = case[0]
        new_structured_cases.append(structured_case)

    raw_cases.extend(new_raw_cases)
    structured_cases.extend(new_structured_cases)

    write_json(raw_cases, raw_path)
    write_json(structured_cases, structured_path)
    print(f"Case Bank updated! new: {len(new_raw_cases)} , total: {len(raw_cases)} .")
