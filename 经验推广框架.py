"""
经验推广框架 —— 基于用户经验 + 联网属性增强的推理引擎

核心流程：
  用户提供经验事件 → 提取 S-A-O-R → 属性生成 → 入库 → 归一化
  用户请求预测 → 提取 S-A-O → 属性生成（用户库）→ 联网补充属性
   → 属性传播 → LLM 加权推理

关键设计：
  - 用户输入的事件作为"硬依据"，权重 1.0
  - 联网补充的属性与用户经验权重相同，推理只看传播距离
  - 属性归一化确保命名一致，不因用词差异丢失匹配
"""

import json
import os
import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from my_llm import LLM

try:
    from ddgs import DDGS
    SEARCH_AVAILABLE = True
except ImportError:
    try:
        from duckduckgo_search import DDGS
        SEARCH_AVAILABLE = True
    except ImportError:
        SEARCH_AVAILABLE = False


# ============================
# 1. 数据结构
# ============================

@dataclass
class Entity:
    name: str
    attrs: Dict[str, float] = field(default_factory=dict)
    source_weights: Dict[str, float] = field(default_factory=dict)


@dataclass
class Experience:
    subject: Entity
    action: Entity
    object: Entity
    result: Entity


@dataclass
class AttrMatch:
    attr_name: str
    value: float
    source_weight: float = 1.0
    source_attr: str = ""
    propagation_distance: float = 0.0
    confidence: float = 1.0


# ============================
# 2. 知识库
# ============================

class KnowledgeBase:
    def __init__(self):
        self.entities: Dict[str, Entity] = {}
        self.cooccurrence: Dict[Tuple[str, str], float] = {}
        self.attr_entities: Dict[str, set] = defaultdict(set)

    def register(self, name: str, attrs: Dict[str, float],
                 source_weight: float = 1.0) -> Entity:
        if name not in self.entities:
            self.entities[name] = Entity(name=name)
        for k, v in attrs.items():
            self.entities[name].attrs[k] = v
            old_w = self.entities[name].source_weights.get(k, 0)
            self.entities[name].source_weights[k] = max(old_w, source_weight)
            self.attr_entities[k].add(name)
        return self.entities[name]

    def update_cooccurrence(self):
        self.cooccurrence.clear()
        cooccur_counts = defaultdict(int)
        attr_counts = defaultdict(int)
        for entity in self.entities.values():
            attrs = list(entity.attrs.keys())
            for a in attrs:
                attr_counts[a] += 1
            for i in range(len(attrs)):
                for j in range(i + 1, len(attrs)):
                    pair = tuple(sorted([attrs[i], attrs[j]]))
                    cooccur_counts[pair] += 1
        for (a1, a2), count in cooccur_counts.items():
            p1 = count / attr_counts[a1] if attr_counts[a1] > 0 else 0
            p2 = count / attr_counts[a2] if attr_counts[a2] > 0 else 0
            self.cooccurrence[(a1, a2)] = min(p1, p2)

    def propagate(self, query_attrs: Dict[str, float],
                  source_weights: Dict[str, float] = None,
                  max_distance: float = 0.3,
                  max_results: int = 15) -> List[AttrMatch]:
        results = []
        seen = set()
        source_weights = source_weights or {}

        for q_attr, q_val in query_attrs.items():
            sw = source_weights.get(q_attr, 1.0)
            if q_attr not in seen:
                results.append(AttrMatch(
                    attr_name=q_attr, value=q_val,
                    source_weight=sw,
                    propagation_distance=0.0,
                    confidence=sw
                ))
                seen.add(q_attr)

            for (a1, a2), prob in self.cooccurrence.items():
                if q_attr in (a1, a2):
                    other = a2 if q_attr == a1 else a1
                    if other not in seen:
                        dist = 1.0 - prob
                        if dist <= max_distance:
                            results.append(AttrMatch(
                                attr_name=other,
                                value=prob * q_val,
                                source_weight=sw,
                                source_attr=q_attr,
                                propagation_distance=dist,
                                confidence=prob * sw
                            ))
                            seen.add(other)

        results.sort(key=lambda m: (m.propagation_distance, -m.confidence))
        return results[:max_results]


# ============================
# 3. 属性归一化
# ============================

class AttributeNormalizer:
    def __init__(self, llm: LLM):
        self.llm = llm
        self.synonym_map: Dict[str, str] = {}

    NORMALIZE_PROMPT = """你是一个属性名标准化专家。
请将以下属性名中含义相同或高度相似的归为一组。

合并规则：
  - "有X"和"X"视为相同
  - "X能力"和"X"视为相同
  - "X性"和"X"视为相同
  - "- X"开头的去掉前缀 "- "后参与合并
  - 只合并含义相同的，不要过度合并
  - 选择最简洁准确的为标准名

属性名列表：
{attr_list}

输出格式（严格 JSON）：
{{
  "groups": [
    {{
      "canonical": "标准名称",
      "synonyms": ["同义词1", "同义词2"]
    }}
  ]
}}"""

    def normalize(self, kb: KnowledgeBase) -> List[Tuple[str, str]]:
        all_attrs = set()
        for entity in kb.entities.values():
            all_attrs.update(entity.attrs.keys())
        if len(all_attrs) < 2:
            return []

        attr_list = "\n".join(f"  - {a}" for a in sorted(all_attrs))
        prompt = self.NORMALIZE_PROMPT.format(attr_list=attr_list)
        resp = self.llm.ask([{"role": "user", "content": prompt}])

        try:
            cleaned = re.search(r"\{.*\}", resp, re.DOTALL)
            if not cleaned:
                return []
            data = json.loads(cleaned.group(0))
        except (json.JSONDecodeError, KeyError):
            return []

        changes = self._clean_prefixes(kb)
        for group in data.get("groups", []):
            canonical = group["canonical"]
            for syn in group.get("synonyms", []):
                syn = syn.strip()
                if syn != canonical and syn in all_attrs:
                    self.synonym_map[syn] = canonical
                    changes.append((syn, canonical))

        if changes:
            self._apply(kb)
        return changes

    def _clean_prefixes(self, kb: KnowledgeBase) -> List[Tuple[str, str]]:
        changes = []
        for entity in kb.entities.values():
            for old_name in list(entity.attrs.keys()):
                new_name = old_name
                if new_name.startswith("- "):
                    new_name = new_name[2:]
                new_name = re.sub(r'^\d+\.\s*', '', new_name)
                new_name = new_name.strip()
                if new_name != old_name:
                    self.synonym_map[old_name] = new_name
                    changes.append((old_name, new_name))
        return changes

    def _apply(self, kb: KnowledgeBase):
        for entity in kb.entities.values():
            merged_attrs: Dict[str, float] = {}
            merged_weights: Dict[str, float] = {}
            for old_name, value in entity.attrs.items():
                cur = old_name
                depth = 0
                while cur in self.synonym_map and depth < 10:
                    cur = self.synonym_map[cur]
                    depth += 1
                existing = merged_attrs.get(cur, 0)
                merged_attrs[cur] = max(existing, value)
                old_w = merged_weights.get(cur, 0)
                merged_weights[cur] = max(old_w,
                    entity.source_weights.get(old_name, 1.0))
            entity.attrs = merged_attrs
            entity.source_weights = merged_weights

        kb.attr_entities.clear()
        for name, entity in kb.entities.items():
            for attr in entity.attrs:
                kb.attr_entities[attr].add(name)
        kb.update_cooccurrence()


# ============================
# 4. LLM 提取层
# ============================

class Extractor:
    def __init__(self, llm: LLM):
        self.llm = llm

    SAO_PROMPT = """你是一个场景分析专家。
请从以下描述中提取：主体(Subject)、动作(Action)、客体(Object)。

格式要求（严格 JSON）：
{{
  "subject": "主体名称",
  "action": "动作名称",
  "object": "客体名称"
}}

描述：{text}"""

    SAOR_PROMPT = """你是一个事件分析专家。
请从以下描述中提取四元组：主体(Subject)、动作(Action)、客体(Object)、结果(Result)。

格式要求（严格 JSON）：
{{
  "subject": "主体名称",
  "action": "动作名称",
  "object": "客体名称",
  "result": "结果名称"
}}

描述：{text}"""

    ATTR_PROMPT = """你是一个属性分析专家。
请为"{entity_name}"列举最重要的属性（最多5个）。

格式：每一行写一个"属性名: 数值(0~1)"

要求：
- 属性要具体有区分力
- 数值0~1反映该属性的显著程度
- 如果以下已有属性中有适用的，请优先使用（保持命名一致）：
{known_attrs}

当前场景关联：{context}"""

    WEB_ATTR_PROMPT = """你是一个属性提取专家。
根据以下关于"{entity_name}"的网络信息，提取最重要的属性（最多3个）。

格式：每一行写一个"属性名: 数值(0~1)"

要求：
- 从网络信息中提取关键特征
- 属性要具体有区分力
- 数值0~1反映该属性的显著程度

网络信息：
{web_info}"""

    def _clean_json(self, text: str) -> str:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        return match.group(0) if match else text

    def extract_sao(self, text: str) -> Tuple[str, str, str]:
        prompt = self.SAO_PROMPT.format(text=text)
        resp = self.llm.ask([{"role": "user", "content": prompt}])
        try:
            data = json.loads(self._clean_json(resp))
            return data["subject"], data["action"], data["object"]
        except (json.JSONDecodeError, KeyError):
            return "未知", "未知", "未知"

    def extract_saor(self, text: str) -> Tuple[str, str, str, str]:
        prompt = self.SAOR_PROMPT.format(text=text)
        resp = self.llm.ask([{"role": "user", "content": prompt}])
        try:
            data = json.loads(self._clean_json(resp))
            return (data["subject"], data["action"],
                    data["object"], data["result"])
        except (json.JSONDecodeError, KeyError):
            return "未知", "未知", "未知", "未知"

    def generate_attrs(self, entity_name: str,
                       context: str = "",
                       known_attrs: List[str] = None,
                       max_attrs: int = 5) -> Dict[str, float]:
        known_str = ""
        if known_attrs:
            known_str = ", ".join(known_attrs[:30])
        prompt = self.ATTR_PROMPT.format(
            entity_name=entity_name, context=context,
            known_attrs=known_str or "（无）",
        )
        resp = self.llm.ask([{"role": "user", "content": prompt}])
        return self._parse_attrs(resp, max_attrs)

    def generate_attrs_from_web(self, entity_name: str,
                                web_info: str,
                                max_attrs: int = 3) -> Dict[str, float]:
        if not web_info.strip():
            return {}
        prompt = self.WEB_ATTR_PROMPT.format(
            entity_name=entity_name, web_info=web_info[:800]
        )
        resp = self.llm.ask([{"role": "user", "content": prompt}])
        return self._parse_attrs(resp, max_attrs)

    def _parse_attrs(self, text: str, max_attrs: int) -> Dict[str, float]:
        attrs = {}
        for line in text.strip().split("\n"):
            if ":" in line:
                parts = line.split(":", 1)
                name = parts[0].strip()
                try:
                    value = float(parts[1].strip())
                    if 0.0 <= value <= 1.0:
                        attrs[name] = value
                except ValueError:
                    pass
                if len(attrs) >= max_attrs:
                    break
        return attrs


# ============================
# 5. 联网属性搜索
# ============================

class WebSearcher:
    """
    联网获取实体属性，作为用户经验的补充依据。

    使用 DuckDuckGo 搜索引擎（免费，无需 API Key）。
    网络不可用时静默降级，不影响框架正常运行。

    为控制 API 成本，每次搜索最多返回 2 条结果、3 个属性。
    """

    def __init__(self, llm: LLM, timeout: int = 5):
        self.llm = llm
        self.extractor = Extractor(llm)
        self.timeout = timeout
        self._ddgs = None
        if SEARCH_AVAILABLE:
            try:
                self._ddgs = DDGS(timeout=self.timeout)
            except Exception:
                self._ddgs = None

    def search_entity_info(self, entity_name: str,
                           context: str = "",
                           max_results: int = 2) -> str:
        if not self._ddgs:
            return ""
        try:
            query = f"{entity_name} {context[:40]} 特征 习性"
            results = list(self._ddgs.text(
                query, max_results=max_results))
            snippets = [r.get("body", "") for r in results if r.get("body")]
            return "\n".join(snippets) if snippets else ""
        except Exception:
            return ""

    def get_web_attributes(self, entity_name: str,
                           context: str = "",
                           known_attrs: List[str] = None,
                           max_attrs: int = 3) -> Dict[str, float]:
        """联网获取属性（最多 max_attrs 个），失败时返回空字典。"""
        if not self._ddgs:
            return {}
        web_info = self.search_entity_info(entity_name, context)
        if not web_info:
            return {}
        attrs = self.extractor.generate_attrs_from_web(
            entity_name, web_info, max_attrs
        )
        return attrs


# ============================
# 6. Agent 主类
# ============================

class ExperienceAgent:
    """
    经验推广 Agent

    用法：
      agent = ExperienceAgent(llm)
      agent.learn("用户输入的事件描述")
      agent.learn("另一个事件描述")
      result = agent.predict("用户输入的预测任务")

    知识库自动持久化：
      - learn / learn_from_file 后自动保存到 knowledge.json
      - 下次创建 Agent 时自动从 knowledge.json 恢复
    """

    KNOWLEDGE_FILE = "knowledge.json"

    def __init__(self, llm: LLM, knowledge_dir: str = None):
        self.llm = llm
        self.kb = KnowledgeBase()
        self.extractor = Extractor(llm)
        self.normalizer = AttributeNormalizer(llm)
        self.searcher = WebSearcher(llm)
        self.experiences: List[Experience] = []
        self.user_attr_weight = 1.0
        self.web_attr_weight = 1.0
        self.max_propagation_dist = 0.30
        self._knowledge_dir = knowledge_dir or os.path.dirname(os.path.abspath(__file__))
        self._load_state()

    def learn(self, text: str):
        """
        从用户提供的事件描述中学习经验。

        参数：
          text: 事件描述字符串（如"在丛林里遇到老虎，老虎扑过来咬伤了我"）

        学习后自动保存知识库到 knowledge.json。
        """
        s, a, o, r = self.extractor.extract_saor(text)
        context = f"事件：{text}"
        known_attr_names = list(set(
            a for pair in self.kb.cooccurrence for a in pair
        ))

        s_attrs = self.extractor.generate_attrs(s, context, known_attr_names)
        a_attrs = self.extractor.generate_attrs(a, context, known_attr_names)
        o_attrs = self.extractor.generate_attrs(o, context, known_attr_names)
        r_attrs = self.extractor.generate_attrs(r, context, known_attr_names)

        self.kb.register(s, s_attrs, source_weight=self.user_attr_weight)
        self.kb.register(a, a_attrs, source_weight=self.user_attr_weight)
        self.kb.register(o, o_attrs, source_weight=self.user_attr_weight)
        self.kb.register(r, r_attrs, source_weight=self.user_attr_weight)
        self.kb.update_cooccurrence()
        self.normalizer.normalize(self.kb)

        self.experiences.append(Experience(
            subject=self.kb.entities[s],
            action=self.kb.entities[a],
            object=self.kb.entities[o],
            result=self.kb.entities[r],
        ))

        self._save_state()

    def learn_from_file(self, filepath: str) -> str:
        """从文档文件中读取并学习经验。"""
        base = os.path.dirname(os.path.abspath(__file__))
        candidates = [filepath, os.path.join(base, filepath)]
        if "." not in os.path.basename(filepath):
            for ext in (".md", ".txt", ".json", ".yaml", ".yml"):
                candidates += [filepath + ext, os.path.join(base, filepath + ext)]
        for p in candidates:
            if os.path.isfile(p):
                filepath = p
                break
        if not os.path.isfile(filepath):
            return f"文件未找到：{filepath}"

        try:
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read()
        except Exception as e:
            return f"读取文件失败：{e}"

        if not content.strip():
            return "文件内容为空。"

        extraction_prompt = f"""从以下文档内容中提取训练事件。
每条事件应为一段完整的经历或事实描述，一行一条。

要求：
- 提取有具体主体、行为、对象、结果的事件
- 不要添加原文没有的信息
- 如果原文没有合适的事件，输出：无

文档内容：
{content[:3000]}

提取的事件："""
        resp = self.llm.ask([{"role": "user", "content": extraction_prompt}])
        lines = [l.strip().lstrip("- ").lstrip("* ") for l in resp.strip().split("\n")
                 if l.strip() and l.strip() != "无"]
        if not lines:
            return "未从文档中提取到训练事件。"

        count = 0
        for line in lines:
            if len(line) > 5:
                self.learn(line)
                count += 1

        return f"✓ 已从文档学习 {count} 条经验（共 {len(self.experiences)} 条）"

    def _prepare_prediction(self, text: str, verbose: bool = False) -> Tuple[str, str, str, str, str, bool]:
        """提取 S-A-O，生成属性，联网搜索，合并传播，构建 prompt。
        返回 (s, a, o, prompt, evidence, has_web)"""
        if verbose:
            print("  ▶ 提取场景元素...", end="", flush=True)
        s, a, o = self.extractor.extract_sao(text)
        if verbose:
            print(f"\r  ✓ 场景元素：{s} → {a} → {o}", flush=True)

        context = f"场景：{text}"
        known_attr_names = list(set(
            a for pair in self.kb.cooccurrence for a in pair
        ))

        if verbose:
            print("  ▶ 分析特征（主体/动作/客体）...", flush=True)
        s_attrs = self.extractor.generate_attrs(s, context, known_attr_names)
        if verbose:
            print(f"    · {s}：{list(s_attrs.keys())[:3]}...", flush=True)
        a_attrs = self.extractor.generate_attrs(a, context, known_attr_names)
        if verbose:
            print(f"    · {a}：{list(a_attrs.keys())[:3]}...", flush=True)
        o_attrs = self.extractor.generate_attrs(o, context, known_attr_names)
        if verbose:
            print(f"    · {o}：{list(o_attrs.keys())[:3]}...", flush=True)

        if verbose:
            print("  ▶ 联网搜索补充知识...", flush=True)
        s_web_attrs = self.searcher.get_web_attributes(s, context, known_attr_names)
        if verbose:
            print(f"    · {s} 联网完毕（{len(s_web_attrs)} 条）", flush=True)
        a_web_attrs = self.searcher.get_web_attributes(a, context, known_attr_names)
        if verbose:
            print(f"    · {a} 联网完毕（{len(a_web_attrs)} 条）", flush=True)
        o_web_attrs = self.searcher.get_web_attributes(o, context, known_attr_names)
        if verbose:
            print(f"    · {o} 联网完毕（{len(o_web_attrs)} 条）", flush=True)

        elem_data = [
            ("主体", s, s_attrs, s_web_attrs),
            ("动作", a, a_attrs, a_web_attrs),
            ("客体", o, o_attrs, o_web_attrs),
        ]

        all_matches = []
        has_web = False

        if verbose:
            print("  ▶ 属性传播匹配...（数值=显著度 0~1，1.0=极显著）", flush=True)
        for role, name, user_attrs, web_attrs in elem_data:
            merged_attrs = dict(user_attrs)
            merged_weights = {k: self.user_attr_weight for k in user_attrs}
            for k, v in web_attrs.items():
                if k not in merged_attrs:
                    merged_attrs[k] = v
                    merged_weights[k] = self.web_attr_weight

            matches = self.kb.propagate(
                merged_attrs,
                source_weights=merged_weights,
                max_distance=self.max_propagation_dist
            )
            all_matches.append((role, name, matches))
            if verbose:
                print(f"    · {role} {name}：{len(matches)} 条匹配", flush=True)
            if web_attrs:
                has_web = True

        lines = []
        lines.append("（属性值=显著度 0~1：1.0=极显著，0.5=中等显著，0.0=不显著）")
        for role, name, matches in all_matches:
            lines.append(f"\n[{role}: {name}]")
            seen = set()
            for m in matches:
                if m.attr_name in seen:
                    continue
                seen.add(m.attr_name)
                if m.propagation_distance == 0:
                    lines.append(
                        f"  - {m.attr_name}={m.value:.2f}（直接）")
                else:
                    lines.append(
                        f"  - {m.attr_name}={m.value:.2f}"
                        f"（←{m.source_attr}，距离={m.propagation_distance:.2f}）")

        evidence = "\n".join(lines)

        common_rules = """- 距离=0.00 的属性为直接匹配，完全可靠
- 距离>0 的属性通过共现传播推断，距离越小越可靠
- 距离>0.30 的属性不可靠，不予采纳"""

        after_source = ""
        if len(self.experiences) > 0:
            after_source += f"（已有 {len(self.experiences)} 条用户经验）"

        prompt = f"""你是一个经验推理专家。

【知识库状态】
- 用户提供的事件数：{len(self.experiences)}
- 已知实体数：{len(self.kb.entities)}

【待预测场景】
{text}

【关键元素】
- 主体：{s}
- 动作：{a}
- 客体：{o}

【属性依据】
以下属性来自知识库和联网搜索{after_source}：

{evidence}

【推理规则】
{common_rules}

请基于以上分析回答用户。先列出关键属性分析结果（包含显著度数值），然后基于这些数据用自然友好的语气给出建议或预测。"""
        return s, a, o, prompt, evidence, has_web

    def predict(self, text: str, verbose: bool = False) -> str:
        """
        对给定场景进行预测。

        流程：
          1. 提取 S-A-O
          2. 从用户知识库生成属性（权重 1.0）
          3. 联网搜索补充属性（最多 3 个/实体）
          4. 合并属性，进行知识库传播
          5. LLM 加权推理

        参数：
          text: 待预测的场景描述
          verbose: 是否显示思考过程

        返回：
          包含属性和推理过程的预测文本
        """
        _, _, _, prompt, _, _ = self._prepare_prediction(text, verbose=verbose)
        if verbose:
            print("  ▶ 生成建议...", flush=True)
        return self.llm.ask([{"role": "user", "content": prompt}])

    def predict_stream(self, text: str, verbose: bool = False):
        """
        对流式预测，逐 token 产出 LLM 推理结果。

        参数：
          text: 待预测的场景描述
          verbose: 是否显示思考过程

        产出：
          逐个 token 的文本片段
        """
        _, _, _, prompt, _, _ = self._prepare_prediction(text, verbose=verbose)
        if verbose:
            print("  ▶ 生成建议...", flush=True)
        for chunk in self.llm.ask_stream([{"role": "user", "content": prompt}]):
            yield chunk

    def _state_path(self) -> str:
        return os.path.join(self._knowledge_dir, self.KNOWLEDGE_FILE)

    def _save_state(self):
        path = self._state_path()
        data = {
            "entities": {
                name: {
                    "attrs": dict(e.attrs),
                    "source_weights": dict(e.source_weights),
                }
                for name, e in self.kb.entities.items()
            },
            "experiences": [
                {"subject": e.subject.name, "action": e.action.name,
                 "object": e.object.name, "result": e.result.name}
                for e in self.experiences
            ],
            "synonym_map": dict(self.normalizer.synonym_map),
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def _load_state(self):
        path = self._state_path()
        if not os.path.isfile(path):
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            return

        for name, edata in data.get("entities", {}).items():
            e = Entity(name=name, attrs=edata["attrs"],
                       source_weights=edata.get("source_weights", {}))
            self.kb.entities[name] = e
            for attr in e.attrs:
                self.kb.attr_entities[attr].add(name)

        self.kb.update_cooccurrence()
        self.normalizer.synonym_map = data.get("synonym_map", {})

        for exp in data.get("experiences", []):
            self.experiences.append(Experience(
                subject=self.kb.entities[exp["subject"]],
                action=self.kb.entities[exp["action"]],
                object=self.kb.entities[exp["object"]],
                result=self.kb.entities[exp["result"]],
            ))

    def show_knowledge(self) -> str:
        """返回当前知识库状态文本"""
        lines = []
        lines.append(f"实体数：{len(self.kb.entities)}")
        lines.append(f"经验数：{len(self.experiences)}")
        lines.append(f"归一化映射：{len(self.normalizer.synonym_map)} 条")
        lines.append("")
        lines.append("--- 实体属性 ---")
        for name, ent in self.kb.entities.items():
            w_str = ""
            if any(w != 1.0 for w in ent.source_weights.values()):
                w_str = " [含网络来源]"
            lines.append(f"  {name}: {ent.attrs}{w_str}")
        if self.kb.cooccurrence:
            lines.append("")
            lines.append(f"--- 属性共现（共 {len(self.kb.cooccurrence)} 对）---")
            for (a1, a2), prob in sorted(
                    self.kb.cooccurrence.items(),
                    key=lambda x: -x[1])[:6]:
                lines.append(f"  {a1} ⟷ {a2}: {prob:.0%}")
        return "\n".join(lines)


if __name__ == "__main__":
    print("经验推广框架 —— 请使用专用 Agent 运行交互（如：基于经验推广框架的用户交互agent.py）")
