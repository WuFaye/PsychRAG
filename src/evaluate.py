
import json
from collections import defaultdict

# --------------- info path ---------------
DRUGCLASS_PATH = 'druginfo/DrugClass.json'
CONTRA_PATH = 'druginfo/contraindications.json'
# -----------------------------------------

# load data
with open(DRUGCLASS_PATH, 'r', encoding='utf-8') as f:
    drug_db = json.load(f)

with open(CONTRA_PATH, 'r', encoding='utf-8') as f:
    contra_db = json.load(f)

# convenience duplication rules
duplication_rules = contra_db.get("duplication_rules", {})
forbidden_duplicates = duplication_rules.get("forbidden_duplicates", [])
forbidden_pairs = duplication_rules.get("forbidden_pairs", [])
allowed_overlap_pairs = duplication_rules.get("allowed_overlap_pairs", [])

def enhance_drug_db(drug_db, contra_db):

    attr = contra_db.get("drug_attributes", {})
    # collect class->drug mapping from contra_db drug_classes if present
    class_to_drugs = {}
    for cls, names in contra_db.get("drug_classes", {}).items():
        class_to_drugs.setdefault(cls, set()).update(names)

    # also attempt to use drug_db['drug_classes'] if that structure exists (some projects keep a mapping)
    if isinstance(drug_db, dict):
        # If top-level contains mapping drug->info, process each
        for drug_name, info in list(drug_db.items()):
            # If info is dict, preserve fields; else create dict
            if not isinstance(info, dict):
                drug_db[drug_name] = {"drug_classes": []}
            # ensure drug_classes exists
            classes = drug_db[drug_name].get("drug_classes", [])
            if not classes:
                # infer from contra_db class lists and class_to_drugs
                inferred = [c for c, names in class_to_drugs.items() if drug_name in names]
                drug_db[drug_name]["drug_classes"] = inferred
            # primary_tag / sub_tag
            if "primary_tag" not in drug_db[drug_name]:
                drug_db[drug_name]["primary_tag"] = drug_db[drug_name]["drug_classes"][0] if drug_db[drug_name].get("drug_classes") else "UNKNOWN"
            if "sub_tag" not in drug_db[drug_name]:
                drug_db[drug_name]["sub_tag"] = drug_db[drug_name].get("primary_tag", "UNKNOWN")
            # qtc risk
            if drug_name in attr.get("qtc_high_risk", []):
                drug_db[drug_name]["qtc_risk"] = "high"
            elif drug_name in attr.get("qtc_moderate_risk", []):
                drug_db[drug_name]["qtc_risk"] = "moderate"
            else:
                drug_db[drug_name]["qtc_risk"] = "low"
            # anticholinergic burden
            if drug_name in attr.get("strong_anticholinergic", []):
                drug_db[drug_name]["anticholinergic_burden"] = 3
            elif drug_name in attr.get("moderate_anticholinergic", []):
                drug_db[drug_name]["anticholinergic_burden"] = 2
            else:
                drug_db[drug_name]["anticholinergic_burden"] = 0
            # cns depressant
            drug_db[drug_name]["cns_depressant"] = bool(set(drug_db[drug_name].get("drug_classes", [])) & set(attr.get("cns_depressant", [])))
            # dopamine flags
            drug_db[drug_name]["dopamine_antagonist"] = False
            da = attr.get("dopamine_antagonists")
            if da:
                if isinstance(da, dict):
                    if drug_name in da.get("examples", []):
                        drug_db[drug_name]["dopamine_antagonist"] = True
                    if da.get("class") and da.get("class") in drug_db[drug_name].get("drug_classes", []):
                        drug_db[drug_name]["dopamine_antagonist"] = True
                elif isinstance(da, list):
                    if drug_name in da:
                        drug_db[drug_name]["dopamine_antagonist"] = True
            drug_db[drug_name]["dopamine_agonist"] = drug_name in attr.get("dopamine_agonists", [])
    return drug_db

drug_db = enhance_drug_db(drug_db, contra_db)

# ---------------- scoring ----------------

def calculate_score(ground_truth, recommendation, alpha=0.5, beta=0.5, concomitant_medications=None):
    """
    - ground_truth: list of drug names
    - recommendation: list of recommended drug names 
    - concomitant_medications: optional list of non-psych meds to include in rationality checks (but not precision/coverage)
    return dict : composite_score, coverage_score, precision_score, rationality_penalty, penalty_details
    """

    gt_main_tags = set()
    for d in ground_truth:
        if d in drug_db:
            gt_main_tags.add(drug_db[d].get("primary_tag"))
    rec_main_tags = set()
    for d in recommendation:
        if d in drug_db:
            rec_main_tags.add(drug_db[d].get("primary_tag"))
    coverage_score = (len(gt_main_tags & rec_main_tags) / len(gt_main_tags)) if gt_main_tags else 0.0

    # precision: primary+sub pair match 
    gt_primary_sub = set()
    for d in ground_truth:
        if d in drug_db:
            gt_primary_sub.add((drug_db[d].get("primary_tag"), drug_db[d].get("sub_tag")))
    matched = 0
    rec_len = max(len(recommendation), 1)
    for rec in recommendation:
        if rec not in drug_db:
            continue
        pair = (drug_db[rec].get("primary_tag"), drug_db[rec].get("sub_tag"))
        if pair in gt_primary_sub:
            matched += 1
    precision_score = matched / rec_len

    # rationality penalties 
    rationality_penalty, penalty_details = calculate_rationality_penalty(recommendation, concomitant_medications)

    composite_score = coverage_score + alpha * precision_score - beta * rationality_penalty

    return {
        "composite_score": round(composite_score, 6),
        "coverage_score": round(coverage_score, 6),
        "precision_score": round(precision_score, 6),
        "rationality_penalty": round(rationality_penalty, 6),
        "penalty_details": penalty_details
    }

# ---------------- rationality / rule checks ----------------

def calculate_rationality_penalty(recommendation, concomitant_medications=None):
    """
    计算 rationality penalty，返回 (total_penalty, penalty_details)
    concomitant_medications: optional list of non-psych meds to include in interaction checks
    """
    penalty_details = {
        "absolute_contraindications": 0.0,
        "major_risks": 0.0,
        "moderate_risks": 0.0,
        "duplication_penalty": 0.0,
        "metabolic_interactions": [],
        "explicit_pair_violations": {},
        "explicit_pair_penalty": 0.0
    }

    rec_drugs = [d for d in recommendation if d in drug_db]
    # include concomitant meds for interaction checks (but do not count them toward precision/coverage)
    concomitant = [d for d in (concomitant_medications or []) if d in drug_db]
    rec_for_checks = list(set(rec_drugs + concomitant))

    # 1) absolute contraindications
    abs_count = check_absolute_contraindications(rec_for_checks)
    abs_pen = abs_count * contra_db.get("contraindication_levels", {}).get("absolute", {}).get("penalty_score", 1.0)
    penalty_details["absolute_contraindications"] = abs_pen

    # 2) major risks (rules 5-9 etc.)
    major_count, major_examples = check_major_risks(rec_for_checks)
    major_pen = major_count * contra_db.get("contraindication_levels", {}).get("major", {}).get("penalty_score", 0.5)
    penalty_details["major_risks"] = major_pen
    penalty_details["major_examples"] = major_examples

    # 3) moderate risks
    moderate_count, moderate_examples = check_moderate_risks(rec_for_checks)
    moderate_pen = moderate_count * contra_db.get("contraindication_levels", {}).get("moderate", {}).get("penalty_score", 0.2)
    penalty_details["moderate_risks"] = moderate_pen
    penalty_details["moderate_examples"] = moderate_examples

    # 4) duplication (supports allowed_overlap_pairs)
    dup_pen = check_duplication(rec_drugs)  # duplication only among recommendations
    penalty_details["duplication_penalty"] = dup_pen

    # 5) metabolic_interactions (drug metabolism / CYP inhibition-avoid combos)
    meta_pen, meta_details = check_metabolic_interactions(rec_for_checks)
    penalty_details["metabolic_interactions"] = meta_details

    # 6) explicit class-pair violations across absolute/major/moderate
    pair_penalty, pair_details = check_explicit_class_pairs(rec_for_checks)
    penalty_details["explicit_pair_violations"] = pair_details
    penalty_details["explicit_pair_penalty"] = round(pair_penalty, 4)
    if pair_details.get("absolute"):
        penalty_details["absolute_contraindications"] += len(pair_details["absolute"]) * contra_db.get("contraindication_levels", {}).get("absolute", {}).get("penalty_score", 1.0)
    if pair_details.get("major"):
        penalty_details["major_risks"] += len(pair_details["major"]) * contra_db.get("contraindication_levels", {}).get("major", {}).get("penalty_score", 0.5)
    if pair_details.get("moderate"):
        penalty_details["moderate_risks"] += len(pair_details["moderate"]) * contra_db.get("contraindication_levels", {}).get("moderate", {}).get("penalty_score", 0.2)

    total_penalty = penalty_details["absolute_contraindications"] + penalty_details["major_risks"] + penalty_details["moderate_risks"] + dup_pen + meta_pen + pair_penalty
    # note: meta_pen is included in total_penalty; we also show details list
    penalty_details["total_penalty_computed"] = round(total_penalty, 4)
    return total_penalty, penalty_details

def check_absolute_contraindications(rec_drugs):
    """
    check absolute.pairs
    """
    violations = 0
    absolute_pairs = contra_db.get("contraindication_levels", {}).get("absolute", {}).get("pairs", [])
    for pair in absolute_pairs:
        a_cls = pair.get("drug_class_a")
        b_cls = pair.get("drug_class_b")
        if not a_cls or not b_cls:
            continue
        a_drugs = [d for d in rec_drugs if a_cls in drug_db[d].get("drug_classes", [])]
        b_drugs = [d for d in rec_drugs if b_cls in drug_db[d].get("drug_classes", [])]
        if a_drugs and b_drugs:
            violations += 1
    # also check duplication_rules.forbidden_pairs
    for pair in forbidden_pairs:
        if len(pair) != 2:
            continue
        a_cls, b_cls = pair[0], pair[1]
        a_drugs = [d for d in rec_drugs if a_cls in drug_db[d].get("drug_classes", [])]
        b_drugs = [d for d in rec_drugs if b_cls in drug_db[d].get("drug_classes", [])]
        if a_drugs and b_drugs:
            violations += 1
    return violations

def check_major_risks(rec_drugs):
    """
    check major rules:
    - multiple_qtc_drugs (rule id=5)
    - lithium_interaction (rule id=6..9)
    - lithium_plus_high_qtc_antipsychotic (rule id=10)
    return (violation_count, examples_list)
    """
    violations = 0
    examples = []
    major_rules = contra_db.get("contraindication_levels", {}).get("major", {}).get("rules", [])

    qtc_high_list = set(contra_db.get("drug_attributes", {}).get("qtc_high_risk", []))
    lithium_list = set(contra_db.get("drug_attributes", {}).get("lithium_drugs", []))
    interacting_examples_map = contra_db.get("drug_attributes", {}).get("interacting_classes_examples", {})

    # identify high qtc drugs present
    high_qtc_drugs = [d for d in rec_drugs if (d in qtc_high_list) or (drug_db[d].get("qtc_risk") == "high")]

    for rule in major_rules:
        rtype = rule.get("type")
        if rtype == "multiple_qtc_drugs":
            thr = rule.get("threshold", 2)
            if len(high_qtc_drugs) >= thr:
                violations += 1
                examples.append({"rule": "multiple_qtc_drugs", "drugs": high_qtc_drugs[:], "threshold": thr})
        elif rtype == "lithium_interaction":
            interacting_classes = rule.get("interacting_classes", []) or contra_db.get("drug_attributes", {}).get("lithium_interacting_classes", [])
            interacting_drugs_found = []
            for d in rec_drugs:
                # by explicit example name
                for cls in interacting_classes:
                    if d in interacting_examples_map.get(cls, []):
                        interacting_drugs_found.append(d)
                        break
                # by class membership
                if set(drug_db[d].get("drug_classes", [])) & set(interacting_classes):
                    interacting_drugs_found.append(d)
            lithium_present = [d for d in rec_drugs if d in lithium_list or "Lithium" in drug_db[d].get("drug_classes", [])]
            # unique
            interacting_drugs_found = list(set(interacting_drugs_found))
            if lithium_present and interacting_drugs_found:
                violations += 1
                examples.append({"rule": "lithium_interaction", "lithium": lithium_present, "interactors": interacting_drugs_found, "interacting_classes": interacting_classes})
        elif rtype == "lithium_plus_high_qtc_antipsychotic" or rule.get("id") == 10:
            # lithium + any high-qtc antipsychotic
            lithium_present = [d for d in rec_drugs if d in lithium_list or "Lithium" in drug_db[d].get("drug_classes", [])]
            high_qtc_antipsychotics = []
            for d in rec_drugs:
                if d in qtc_high_list or drug_db[d].get("qtc_risk") == "high":
                    # prefer to ensure it's antipsychotic, but include any high-qtc
                    if "Antipsychotic" in drug_db[d].get("drug_classes", []) or drug_db[d].get("primary_tag") == "Antipsychotic":
                        high_qtc_antipsychotics.append(d)
            if lithium_present and high_qtc_antipsychotics:
                violations += 1
                examples.append({"rule": "lithium_plus_high_qtc_antipsychotic", "lithium": lithium_present, "high_qtc_antipsychotics": high_qtc_antipsychotics})
        else:
            # other major rules
            pass

    return violations, examples

def check_moderate_risks(rec_drugs):
    """
    moderate rules: multiple_cns_depressants, high_anticholinergic_burden, dopamine_agonist_antagonist
    Note: explicit class-pairs in moderate.rules are handled by check_explicit_class_pairs
    """
    violations = 0
    examples = []
    moderate_rules = contra_db.get("contraindication_levels", {}).get("moderate", {}).get("rules", [])

    for rule in moderate_rules:
        rtype = rule.get("type")
        if rtype == "multiple_cns_depressants":
            thr = rule.get("threshold", 3)
            cns_depressants = [d for d in rec_drugs if drug_db[d].get("cns_depressant")]
            if len(cns_depressants) >= thr:
                violations += 1
                examples.append({"rule": "multiple_cns_depressants", "drugs": cns_depressants})
        elif rtype == "high_anticholinergic_burden":
            thr = rule.get("threshold", 3)
            total_burden = sum(drug_db[d].get("anticholinergic_burden", 0) for d in rec_drugs)
            if total_burden >= thr:
                violations += 1
                examples.append({"rule": "high_anticholinergic_burden", "burden": total_burden, "drugs": rec_drugs})
        elif rtype == "dopamine_agonist_antagonist":
            dopamine_agonists = [d for d in rec_drugs if drug_db[d].get("dopamine_agonist")]
            dopamine_antagonists = [d for d in rec_drugs if drug_db[d].get("dopamine_antagonist")]
            if dopamine_agonists and dopamine_antagonists:
                violations += 1
                examples.append({"rule": "dopamine_agonist_antagonist", "agonists": dopamine_agonists, "antagonists": dopamine_antagonists})
        else:
            # explicit class pairs are skipped here (they are handled separately)
            pass

    return violations, examples

def check_explicit_class_pairs(rec_drugs):
    """
    在 absolute/major/moderate 任意层级中查找显式声明的 drug_class_a/drug_class_b 对，并按层级 penalty_score 累计
    返回 (total_pair_penalty, details_dict)
    """
    details = {"absolute": [], "major": [], "moderate": []}
    total_pair_penalty = 0.0
    levels = ["absolute", "major", "moderate"]
    for level in levels:
        lvl = contra_db.get("contraindication_levels", {}).get(level, {})
        penalty_score = lvl.get("penalty_score", 0.0)
        explicit_pairs = []
        if level == "absolute":
            explicit_pairs.extend([ (p.get("drug_class_a"), p.get("drug_class_b"), p.get("description", "")) 
                                    for p in lvl.get("pairs", []) if p.get("drug_class_a") and p.get("drug_class_b") ])
        for r in lvl.get("rules", []):
            if isinstance(r, dict) and r.get("drug_class_a") and r.get("drug_class_b"):
                explicit_pairs.append((r.get("drug_class_a"), r.get("drug_class_b"), r.get("description", "")))
        for a_cls, b_cls, desc in explicit_pairs:
            a_drugs = [d for d in rec_drugs if a_cls in drug_db[d].get("drug_classes", [])] if 'rec_drugs' in locals() else []
            # rec_drugs not defined here, so we will compute based on passed-in rec set in caller; but for safety compute using rec_drugs in outer context
            # Actually, this function expects rec_drugs param; we adapted earlier by passing rec_for_checks into call site.
            # For conversion safety we will assume rec_drugs is passed in properly by caller (see calculate_rationality_penalty)
            # But since rec_drugs isn't parameter here, we'll fallback:
            pass
    # Note: this function is invoked in calculate_rationality_penalty which has rec_for_checks; to avoid confusion we re-implement a local version here:
    return check_explicit_class_pairs_impl()

def check_explicit_class_pairs_impl():
    """
    Implementation used by wrapper: internally reads rec_for_checks from calculate_rationality_penalty
    We will accept rec_for_checks through global fallback: the calculate_rationality_penalty calls this inner impl with rec_for_checks param.
    """
    # This impl is intentionally left unreachable; we instead define a proper version above and call it from calculate_rationality_penalty
    return 0.0, {"absolute": [], "major": [], "moderate": []}

# To avoid the confusion above, we will define a clear version that accepts rec_drugs param and use it in calls:
def check_explicit_class_pairs(rec_drugs):
    details = {"absolute": [], "major": [], "moderate": []}
    total_pair_penalty = 0.0
    levels = ["absolute", "major", "moderate"]
    for level in levels:
        lvl = contra_db.get("contraindication_levels", {}).get(level, {})
        penalty_score = lvl.get("penalty_score", 0.0)
        explicit_pairs = []
        if level == "absolute":
            explicit_pairs.extend([ (p.get("drug_class_a"), p.get("drug_class_b"), p.get("description", "")) 
                                    for p in lvl.get("pairs", []) if p.get("drug_class_a") and p.get("drug_class_b") ])
        for r in lvl.get("rules", []):
            if isinstance(r, dict) and r.get("drug_class_a") and r.get("drug_class_b"):
                explicit_pairs.append((r.get("drug_class_a"), r.get("drug_class_b"), r.get("description", "")))
        for a_cls, b_cls, desc in explicit_pairs:
            a_drugs = [d for d in rec_drugs if a_cls in drug_db[d].get("drug_classes", [])]
            b_drugs = [d for d in rec_drugs if b_cls in drug_db[d].get("drug_classes", [])]
            if a_drugs and b_drugs:
                details[level].append({
                    "pair": (a_cls, b_cls),
                    "a_drugs": a_drugs,
                    "b_drugs": b_drugs,
                    "description": desc
                })
                total_pair_penalty += penalty_score
    return total_pair_penalty, details

def check_duplication(rec_drugs):

    penalty = 0.0
    sub_tag_map = defaultdict(list)
    for d in rec_drugs:
        if d in drug_db:
            sub_tag = drug_db[d].get("sub_tag") or drug_db[d].get("primary_tag", "UNKNOWN")
            sub_tag_map[sub_tag].append(d)
    dup_base = contra_db.get("duplication_penalty", 0.2)
    for sub_tag, drugs in sub_tag_map.items():
        cnt = len(drugs)
        if cnt <= 1:
            continue
        all_pairs_allowed = True
        for i in range(len(drugs)):
            for j in range(i+1, len(drugs)):
                a, b = drugs[i], drugs[j]
                pair_allowed = False
                for allowed in allowed_overlap_pairs:
                    if not isinstance(allowed, (list, tuple)) or len(allowed) != 2:
                        continue
                    # drug-name match
                    if (a == allowed[0] and b == allowed[1]) or (a == allowed[1] and b == allowed[0]):
                        pair_allowed = True
                        break
                    # class-level match
                    a_classes = set(drug_db[a].get("drug_classes", []))
                    b_classes = set(drug_db[b].get("drug_classes", []))
                    if allowed[0] in a_classes and allowed[1] in b_classes:
                        pair_allowed = True
                        break
                if not pair_allowed:
                    all_pairs_allowed = False
                    break
            if not all_pairs_allowed:
                break
        if not all_pairs_allowed:
            penalty += dup_base * (cnt - 1)
    return penalty

def check_metabolic_interactions(rec_drugs):
    """
    check metabolic_interactions.rules(explicit drug pairs)，retuen (penalty_sum, details_list)
    """
    rules = contra_db.get("metabolic_interactions", {}).get("rules", [])
    penalty_score = contra_db.get("metabolic_interactions", {}).get("penalty_score", 0.5)
    details = []
    total_pen = 0.0
    for r in rules:
        a = r.get("drug_a")
        b = r.get("drug_b")
        if not a or not b:
            continue
        if a in rec_drugs and b in rec_drugs:
            details.append({"pair": (a, b), "description": r.get("description", "")})
            total_pen += penalty_score
    return total_pen, details

# ------------------ main / demo ------------------
if __name__ == "__main__":
    # demo: 支持传 concurrent meds
    gt = ["舍曲林", "丙戊酸"]
    rec1 = ["舍曲林", "文拉法辛", "劳拉西泮", "丙戊酸", "阿普唑仑"]  # SSRI+SNRI
    rec2 = ["碳酸锂", "齐拉西酮", "劳拉西泮", "丙戊酸", "阿普唑仑"]  # lithium + ziprasidone (should trigger major)
    rec3 = ["氟伏沙明", "阿戈美拉汀", "劳拉西泮", "丙戊酸", "阿普唑仑"]  # metabolic interaction example
    # concurrent meds (e.g., 降压药)
    concomitant = ["氢氯噻嗪"]
    tests = [
        ("SSRI+SNRI", rec1, None),
        ("Lithium + Ziprasidone", rec2, None),
        ("Fluvoxamine + Agomelatine (metabolic)", rec3, None),
        ("Lithium + Diuretic (via concomitant)", ["碳酸锂", "丙戊酸", "舍曲林", "劳拉西泮", "阿普唑仑"], concomitant)
    ]
    for name, rec, concom in tests:
        print("="*60)
        print("Test:", name)
        out = calculate_score(gt, rec, concomitant_medications=concom)
        print("Recommendation:", rec, "Concomitant:", concom)
        print("composite:", out["composite_score"], "coverage:", out["coverage_score"], "precision:", out["precision_score"], "rationality_penalty:", out["rationality_penalty"])
        print("penalty_details:")
        for k, v in out["penalty_details"].items():
            print(" -", k, ":", v)
        print("\n")
