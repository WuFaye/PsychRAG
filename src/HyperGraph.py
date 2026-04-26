import hashlib
import json
import os
from typing import Dict, List, Set, Optional, Any
from collections import defaultdict
import networkx as nx

try:
    from pyvis.network import Network
    _HAS_PYVIS = True
except Exception:
    _HAS_PYVIS = False


def _norm(text: str) -> str:
    if not isinstance(text, str):
        return text
    return text.strip().lower()


class HyperGraphRAG:
    def __init__(
        self,
        nodeclass: List[tuple] = [("症状节点", "symptom"), ("药物节点", "drug")],
        enable_pyvis: bool = True
    ):
        self.node_class = nodeclass
        # {node_id: {"id", "type", "name", "relations": set(edge_id)}}
        self.node_store: Dict[str, Dict[str, Any]] = {}
        # {edge_id: {"id","nodes": set(node_id), "knowledge": str, "properties": {...}}}
        self.hyperedge_store: Dict[str, Dict[str, Any]] = {}
        self.index: Dict[str, Dict[str, Set[str]]] = {
            self.node_class[0][1]: defaultdict(set),
            self.node_class[1][1]: defaultdict(set)
        }
        self.synonyms: Dict[str, Set[str]] = defaultdict(set)
        self.enable_pyvis = enable_pyvis and _HAS_PYVIS

    def _generate_id(self, content: str) -> str:
        return hashlib.md5(content.encode("utf-8")).hexdigest()[:8]


    def _ensure_node(self, raw_name: str, node_type: str) -> str:
        norm_name = _norm(raw_name)
        existing_ids = self.index[node_type].get(norm_name, set())
        if existing_ids:
            node_id = next(iter(existing_ids))
            node = self.node_store[node_id]
            return node_id

        # new node
        node_id = self._generate_id(f"{node_type}_{raw_name}")
        self.node_store[node_id] = {
            "id": node_id,
            "type": node_type,
            "name": raw_name,
            "relations": set()
        }
        self.index[node_type][_norm(raw_name)].add(node_id)
        return node_id

    def insert_context(self, raw_data: Dict[str, Any], source_id: Optional[str] = None):

        symptom_field, drug_field = self.node_class[0][0], self.node_class[1][0]
        symptoms = raw_data.get(symptom_field, [])
        drugs = raw_data.get(drug_field, [])
        knowledge = raw_data.get("知识", raw_data.get("knowledge", ""))

        symptom_nodes = [self._ensure_node(s, self.node_class[0][1]) for s in symptoms]
        drug_nodes = [self._ensure_node(d, self.node_class[1][1]) for d in drugs]

        edge_id = self._generate_id("|".join(sorted(symptom_nodes + drug_nodes)) + knowledge + (source_id or ""))
        self.hyperedge_store[edge_id] = {
            "id": edge_id,
            "nodes": set(symptom_nodes + drug_nodes),
            "knowledge": knowledge,
            "properties": {
                "symptoms": list(symptoms),
                "drugs": list(drugs),
                "source": source_id
            }
        }

        for nid in symptom_nodes + drug_nodes:
            self.node_store[nid]["relations"].add(edge_id)

        for s in symptoms:
            self.index[self.node_class[0][1]][_norm(s)].add(self._generate_id(f"{self.node_class[0][1]}_{s}"))  # safe-guard but primary mapping already done above


    def add_synonym(self, std_term: str, alias: str):

        self.synonyms[_norm(std_term)].add(_norm(alias))

    def add_synonyms(self, std_term: str, aliases: List[str]):
        for a in aliases:
            self.add_synonym(std_term, a)

    def _expand_with_synonyms(self, term: str) -> Set[str]:

        norm_term = _norm(term)
        expanded = {norm_term}

        if norm_term in self.synonyms:
            expanded.update(self.synonyms[norm_term])

        for std, aliases in self.synonyms.items():
            if norm_term in aliases:
                expanded.add(std)
                expanded.update(aliases)
        return expanded


    def query_by_symptoms(
        self,
        symptoms: List[str],
        fuzzy: bool = True,
        expand_synonyms: bool = True
    ) -> List[Dict[str, Any]]:

        results = []
        seen_edges = set()
        symptom_index = self.index[self.node_class[0][1]]

        query_terms = set()
        for s in symptoms:
            if expand_synonyms:
                query_terms.update(self._expand_with_synonyms(s))
            else:
                query_terms.add(_norm(s))

        if not fuzzy:

            for q in query_terms:
                node_ids = symptom_index.get(q, set())
                for nid in node_ids:
                    for eid in self.node_store[nid]["relations"]:
                        if eid in seen_edges:
                            continue
                        edge = self.hyperedge_store.get(eid)
                        if not edge:
                            continue
                        results.append({
                            "edge_id": eid,
                            "symptoms": edge["properties"].get("symptoms", []),
                            "drugs": edge["properties"].get("drugs", []),
                            "knowledge": edge.get("knowledge", ""),
                            "source": edge["properties"].get("source")
                        })
                        seen_edges.add(eid)
            return results


        for q in query_terms:
            for indexed_sym, node_ids in symptom_index.items():
                if q in indexed_sym or indexed_sym in q:
                    for nid in node_ids:
                        for eid in self.node_store[nid]["relations"]:
                            if eid in seen_edges:
                                continue
                            edge = self.hyperedge_store.get(eid)
                            if not edge:
                                continue
                            results.append({
                                "edge_id": eid,
                                "symptoms": edge["properties"].get("symptoms", []),
                                "drugs": edge["properties"].get("drugs", []),
                                "knowledge": edge.get("knowledge", ""),
                                "source": edge["properties"].get("source")
                            })
                            seen_edges.add(eid)
        return results

    def get_related_knowledge(self, related_nodes: List[str], node_types: Optional[List[str]] = None) -> List[Dict[str, Any]]:

        results = []
        seen_edges = set()
        types = node_types if node_types is not None else list(self.index.keys())

        for raw_name in related_nodes:
            norm_name = _norm(raw_name)
            for t in types:

                expanded = self._expand_with_synonyms(raw_name)
                if not expanded:
                    expanded = {norm_name}
                for key in expanded:
                    node_ids = self.index.get(t, {}).get(key, set())
                    for nid in node_ids:
                        for eid in self.node_store[nid].get("relations", []):
                            if eid in seen_edges:
                                continue
                            edge = self.hyperedge_store.get(eid)
                            if not edge:
                                continue
                            results.append({
                                edge.get("knowledge", "")
                            })
                            seen_edges.add(eid)
        return results


    def to_networkx(self) -> nx.Graph:
        """
        将超图导出为 NetworkX 图：
        - 超边作为单独节点（类型: 'hyperedge'）
        - 症状/药物节点保持原样
        """
        G = nx.Graph()
        # add nodes
        for nid, node in self.node_store.items():
            G.add_node(nid, label=node["name"], type=node["type"])
        for eid, edge in self.hyperedge_store.items():
            G.add_node(eid, label=f"Edge:{eid[:6]}", type="hyperedge", knowledge=edge.get("knowledge", "") )
            for nid in edge["nodes"]:
                if nid in self.node_store:
                    G.add_edge(eid, nid)
        return G

    def visualize_full_graph(self, output_file: str = "medical_kg.html", height: str = "800px", width: str = "100%"):
        """使用 pyvis 导出交互式 HTML（若未安装 pyvis 将提示）"""
        if not self.enable_pyvis:
            print("pyvis 未安装或被禁用，无法生成 HTML 可视化。请安装 pyvis 或将 enable_pyvis=True。")
            return

        nt = Network(height=height, width=width, notebook=False, directed=False)
        nt.toggle_physics(True)
        added_nodes = set()

        # hyperedges first
        for edge_id, edge in self.hyperedge_store.items():
            nt.add_node(edge_id, shape="hexagon", color="#FFD700", size=30,
                        title=edge.get("knowledge", ""), label=f"治疗方案 {edge_id[:6]}")
            added_nodes.add(edge_id)

        for node_id, node in self.node_store.items():
            node_color = "#FFA07A" if node["type"] == "symptom" else "#98FB98"
            shape = "circle" if node["type"] == "symptom" else "square"
            nt.add_node(node_id, label=node["name"], color=node_color, shape=shape, size=25, title=node["name"])
            added_nodes.add(node_id)

        for edge_id, edge in self.hyperedge_store.items():
            for nid in edge.get("nodes", []):
                if nid in self.node_store:
                    nt.add_edge(edge_id, nid, width=2, color="#808080")

        try:
            html = nt.generate_html()
            html = html.replace('<head>', '<head>\n    <meta charset="UTF-8">')
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(html)
            print(f"可视化文件已生成: {os.path.abspath(output_file)}")
        except Exception as e:
            print("生成可视化失败：", str(e))


    def save_to_file(self, path: str):
        obj = {
            "node_class": self.node_class,
            "node_store": {k: {"id": v["id"], "type": v["type"], "name": v["name"], "relations": list(v["relations"])} for k, v in self.node_store.items()},
            "hyperedge_store": {k: {"id": v["id"], "nodes": list(v["nodes"]), "knowledge": v["knowledge"], "properties": v["properties"]} for k, v in self.hyperedge_store.items()},
            "synonyms": {k: list(v) for k, v in self.synonyms.items()}
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)
        print("已保存到", path)

    def load_from_file(self, path: str):
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        self.node_class = obj.get("node_class", self.node_class)
        self.node_store = {k: {"id": v["id"], "type": v["type"], "name": v["name"], "relations": set(v["relations"])} for k, v in obj.get("node_store", {}).items()}
        self.hyperedge_store = {k: {"id": v["id"], "nodes": set(v["nodes"]), "knowledge": v.get("knowledge",""), "properties": v.get("properties", {})} for k, v in obj.get("hyperedge_store", {}).items()}
        # rebuild index
        self.index = {self.node_class[0][1]: defaultdict(set), self.node_class[1][1]: defaultdict(set)}
        for nid, n in self.node_store.items():
            self.index[n["type"]][_norm(n["name"])].add(nid)
        self.synonyms = defaultdict(set, {k: set(v) for k, v in obj.get("synonyms", {}).items()})
        print("已从", path, "加载知识图谱（节点/超边数量：", len(self.node_store), len(self.hyperedge_store), ")")

    def get_all_nodes(self, node_type: str = None) -> List[str]:
        if node_type:
            return [n["name"] for n in self.node_store.values() if n["type"] == node_type]
        return [n["name"] for n in self.node_store.values()]

    def debug_state(self) -> Dict[str, int]:
        return {
            "node_count": len(self.node_store),
            "hyperedge_count": len(self.hyperedge_store),
            "synonym_entries": len(self.synonyms)
        }

# an example usage of the HyperGraphRAG class with medical knowledge graph construction and querying
'''
if __name__ == "__main__":
    kg = HyperGraphRAG()
    kg.add_synonyms("抑郁", ["情绪低落", "情绪低", "重度抑郁发作"])

    examples = [
        {'症状节点': ['重度抑郁发作', '精神病性症状', '幻听'], '药物节点': ['文拉法辛', '阿立哌唑'], '知识': '联合使用文拉法辛和阿立哌唑可改善抑郁和幻听。'},
        {'症状节点': ['幻听', '被害妄想'], '药物节点': ['阿立哌唑'], '知识': '阿立哌唑对幻听和被害妄想有效。'},
        {'症状节点': ['入睡困难', '眠浅易醒'], '药物节点': ['劳拉西泮', '奥沙西泮'], '知识': '劳拉西泮改善入睡困难，奥沙西泮可用于肢体不适时替代。'},
        {'症状节点': ['情绪低落', '自杀观念'], '药物节点': ['碳酸锂', '喹硫平'], '知识': '碳酸锂联合喹硫平对伴强烈自杀观念的抑郁有效。'}
    ]
    for i, ex in enumerate(examples):
        kg.insert_context(ex, source_id=f"example_{i}")

    print("state:", kg.debug_state())

    res = kg.query_by_symptoms(["抑郁"], fuzzy=False)
    print("\n=== 精确/同义词扩展 查询 '抑郁' ===")
    for r in res:
        print("-", r["edge_id"], r["symptoms"], "->", r["drugs"], "|", r["knowledge"])

    res_fuzzy = kg.query_by_symptoms(["抑郁"], fuzzy=True)
    print("\n=== 模糊 查询 '抑郁' ===")
    for r in res_fuzzy:
        print("-", r["edge_id"], r["symptoms"], "->", r["drugs"], "|", r["knowledge"])

    if _HAS_PYVIS:
        kg.visualize_full_graph("example_medical_kg.html")
    else:
        print("pyvis has not been installed (pip install pyvis to enable HTML export)")'''
