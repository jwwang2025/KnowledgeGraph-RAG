"""CoT 思维链推理：Zero-shot / Few-shot / Self-Consistency 多模式推理与 prompt 构建。"""

import re
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from enum import Enum


class CoTMode(Enum):
    """思维链模式"""
    ZERO_SHOT = "zero_shot"
    FEW_SHOT = "few_shot"
    SELF_CONSISTENCY = "self_consistency"
    DIRECT = "direct"


@dataclass
class ReasoningStep:
    """推理步骤"""
    step_num: int = 0
    title: str = ""
    content: str = ""
    evidence: List[str] = field(default_factory=list)
    conclusion: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step_num": self.step_num,
            "title": self.title,
            "content": self.content,
            "evidence": self.evidence,
            "conclusion": self.conclusion
        }


@dataclass
class ReasoningChain:
    """完整推理链"""
    query: str = ""
    mode: CoTMode = CoTMode.ZERO_SHOT
    steps: List[ReasoningStep] = field(default_factory=list)
    intermediate_conclusions: List[str] = field(default_factory=list)
    final_answer: str = ""
    consistency_score: float = 0.0  # 仅 SELF_CONSISTENCY 模式使用
    path_answers: List[str] = field(default_factory=list)
    depth: int = 1
    used_knowledge: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "mode": self.mode.value,
            "steps": [s.to_dict() for s in self.steps],
            "intermediate_conclusions": self.intermediate_conclusions,
            "final_answer": self.final_answer,
            "consistency_score": self.consistency_score,
            "depth": self.depth,
            "used_knowledge": self.used_knowledge
        }

    def to_visualization(self) -> str:
        """生成可视化格式的推理链"""
        lines = [f"🔍 问题: {self.query}", "", "📝 推理过程:"]

        for step in self.steps:
            lines.append(f"  步骤 {step.step_num}: {step.title}")
            lines.append(f"    {step.content}")
            if step.evidence:
                lines.append(f"    📌 证据: {'; '.join(step.evidence[:2])}")
            if step.conclusion:
                lines.append(f"    ✅ 小结: {step.conclusion}")
            lines.append("")

        lines.append(f"🎯 最终答案: {self.final_answer}")
        return "\n".join(lines)


@dataclass
class FewShotExample:
    """Few-shot 示例"""
    question: str
    reasoning_steps: List[str]
    answer: str

    def to_prompt(self) -> str:
        """转换为 prompt 格式"""
        lines = [f"问题: {self.question}"]
        for i, step in enumerate(self.reasoning_steps, 1):
            lines.append(f"推理 {i}: {step}")
        lines.append(f"答案: {self.answer}")
        return "\n".join(lines)


class CoTReasoner:
    """思维链推理器：提供多种 CoT 模式来增强推理能力"""

    ZERO_SHOT_SUFFIX = "\n\n请逐步思考并给出答案："

    DEFAULT_EXAMPLES: List[FewShotExample] = [
        FewShotExample(
            question="人工智能和机器学习有什么区别？",
            reasoning_steps=[
                "首先明确人工智能的定义：使机器具有人类智能的技术",
                "然后明确机器学习的定义：让机器从数据中自动学习的技术",
                "分析两者的关系：机器学习是人工智能的一个子领域",
                "比较应用范围：AI 更广泛，ML 更专注于学习算法"
            ],
            answer="人工智能(AI)是一个广义概念，指使机器具有人类智能的技术；而机器学习(ML)是AI的子领域，专注于让机器从数据中自动学习和改进。简单说，ML是实现AI的一种方法。"
        ),
        FewShotExample(
            question="什么是知识图谱？",
            reasoning_steps=[
                "知识图谱是一种结构化知识表示方法",
                "核心要素：实体、关系、属性",
                "以图的形式存储知识，节点表示实体，边表示关系",
                "常用于增强搜索、问答等应用"
            ],
            answer="知识图谱是一种用图结构表示知识的系统，由实体、关系和属性组成。它以节点表示实体、以边表示实体间关系的方式，将结构化知识存储在图中，广泛应用于智能搜索、问答系统和推荐系统等领域。"
        )
    ]

    _QUERY_TYPE_KEYWORDS = [
        ("定义类", ["什么", "什么是", "定义"]),
        ("事实类", ["谁", "哪里", "什么时候"]),
        ("解释类", ["为什么", "原因"]),
        ("方法类", ["如何", "怎么", "方法"]),
        ("比较类", ["比较", "区别", "不同"]),
    ]

    _ENTITY_PATTERNS = [
        r'[""]([^""]+)[""]',
        r'[「』]([^「」]+)[「」]',
        r'([\u4e00-\u9fa5]{2,})(是谁|是什么|在哪里)',
        r'([\u4e00-\u9fa5]{2,})和([\u4e00-\u9fa5]{2,})',
        r'的([\u4e00-\u9fa5]{2,})(是什么|有什么关系)',
    ]

    def __init__(self, mode: CoTMode = CoTMode.ZERO_SHOT,
                 custom_examples: List[FewShotExample] = None,
                 enable_self_consistency: bool = False,
                 num_paths: int = 3):
        self.mode = mode
        self.examples = custom_examples or self.DEFAULT_EXAMPLES
        self.enable_self_consistency = enable_self_consistency
        self.num_paths = num_paths

    def reason(self, query: str, knowledge_context: str,
               depth: int = 1) -> ReasoningChain:
        """执行思维链推理"""
        chain = ReasoningChain(query=query, mode=self.mode, depth=depth)

        if self.mode == CoTMode.DIRECT:
            return self._direct_reasoning(query, knowledge_context, chain)
        elif self.mode == CoTMode.ZERO_SHOT:
            return self._zero_shot_reasoning(query, knowledge_context, chain, depth)
        elif self.mode == CoTMode.FEW_SHOT:
            return self._few_shot_reasoning(query, knowledge_context, chain, depth)
        elif self.mode == CoTMode.SELF_CONSISTENCY:
            return self._self_consistency_reasoning(query, knowledge_context, chain, depth)

        return chain

    def _direct_reasoning(self, query: str, knowledge: str,
                          chain: ReasoningChain) -> ReasoningChain:
        """直接回答 (无思维链)"""
        chain.final_answer = f"根据知识库信息回答: {query}"
        chain.used_knowledge = [knowledge[:200]] if knowledge else []
        return chain

    def _zero_shot_reasoning(self, query: str, knowledge: str,
                             chain: ReasoningChain, depth: int) -> ReasoningChain:
        """Zero-shot CoT: 基于隐式推理"""
        sub_questions = self._decompose_question(query, depth)

        for i, sub_q in enumerate(sub_questions, 1):
            step = ReasoningStep()
            step.step_num = i
            step.title = self._get_step_title(i)
            step.content = f"分析: {sub_q}"
            step.evidence = [knowledge[:300]] if knowledge else []
            step.conclusion = self._implicit_inference(sub_q, knowledge)

            chain.steps.append(step)
            chain.intermediate_conclusions.append(step.conclusion)

        chain.final_answer = self._synthesize_answer(query, chain.intermediate_conclusions, knowledge)
        chain.used_knowledge = [knowledge[:200]] if knowledge else []
        return chain

    def _few_shot_reasoning(self, query: str, knowledge: str,
                            chain: ReasoningChain, depth: int) -> ReasoningChain:
        """Few-shot CoT: 基于示例推理"""
        example_prompts = [ex.to_prompt() for ex in self.examples[:2]]

        step1 = ReasoningStep()
        step1.step_num = 1
        step1.title = "📖 问题理解"
        step1.content = f"问题类型: {self._classify_query_type(query)}\n关键实体: {self._extract_entities(query)}"
        step1.evidence = example_prompts[:1]
        step1.conclusion = f"这是一个需要综合分析的问题"
        chain.steps.append(step1)

        if knowledge:
            step2 = ReasoningStep()
            step2.step_num = 2
            step2.title = "📚 知识检索"
            step2.content = f"从外部知识库检索到相关信息:\n{knowledge[:400]}"
            step2.evidence = [knowledge[:300]]
            step2.conclusion = f"找到 {len(knowledge)//100 + 1} 条相关信息"
            chain.steps.append(step2)
            chain.used_knowledge.append(knowledge[:200])

        step3 = ReasoningStep()
        step3.step_num = 3
        step3.title = "🔍 推理分析"
        step3.content = self._build_inference_prompt(query, knowledge)
        step3.evidence = chain.used_knowledge
        step3.conclusion = self._simulate_reasoning(query, knowledge, chain.used_knowledge).get("conclusion", "")
        chain.steps.append(step3)
        chain.intermediate_conclusions.append(step3.conclusion)

        step4 = ReasoningStep()
        step4.step_num = 4
        step4.title = "✅ 综合结论"
        step4.content = f"综合以上分析，得出最终答案"
        step4.conclusion = self._synthesize_answer(query, chain.intermediate_conclusions, knowledge)
        chain.final_answer = step4.conclusion
        chain.steps.append(step4)

        return chain

    def _self_consistency_reasoning(self, query: str, knowledge: str,
                                    chain: ReasoningChain, depth: int) -> ReasoningChain:
        """Self-Consistency: 多路径推理取最优"""
        all_answers = []
        path_titles = ["事实检索", "概念解析", "关系分析"]
        path_contents = [
            f"从知识图谱检索事实: {knowledge[:200]}",
            f"理解核心概念: {query}",
            f"分析实体间关系: {knowledge[:200]}",
        ]
        path_evidences = [["三元组信息"], ["定义信息"], ["关系三元组"]]
        path_conclusions = ["确认基础事实", "明确概念内涵", "理清关联脉络"]

        for path_id in range(self.num_paths):
            path_chain = ReasoningChain(query=query, mode=self.mode, depth=depth)

            if path_id < 3:
                path_chain.steps.append(self._create_step(1, path_titles[path_id],
                    path_contents[path_id], path_evidences[path_id], path_conclusions[path_id]))

            path_chain.steps.append(self._create_step(2, "逻辑推演",
                f"基于以上信息进行逻辑推导", [], "形成推理结论"))

            path_answer = self._synthesize_answer(query, [], knowledge)
            path_chain.final_answer = path_answer
            all_answers.append(path_answer)

        chain.path_answers = all_answers
        chain.final_answer = self._select_consistent_answer(all_answers)
        chain.consistency_score = self._calculate_consistency(all_answers)

        chain.steps.append(self._create_step(1, "多路径推理",
            f"生成了 {len(all_answers)} 条推理路径", [], "完成多路径分析"))
        chain.steps.append(self._create_step(2, "一致性选择",
            f"一致性得分: {chain.consistency_score:.2f}", [], f"最终答案: {chain.final_answer}"))

        return chain

    def _create_step(self, num: int, title: str, content: str,
                     evidence: List[str], conclusion: str) -> ReasoningStep:
        """创建推理步骤的辅助方法"""
        step = ReasoningStep()
        step.step_num = num
        step.title = title
        step.content = content
        step.evidence = evidence
        step.conclusion = conclusion
        return step

    def _decompose_question(self, query: str, depth: int) -> List[str]:
        """分解问题为子问题"""
        sub_questions = []

        if depth >= 1:
            sub_questions.append(f"这个问题是什么类型？({self._classify_query_type(query)})")
            entities = self._extract_entities(query)
            if entities:
                sub_questions.append(f"问题涉及哪些关键实体？({', '.join(entities[:3])})")

        if depth >= 2:
            sub_questions.append(f"如何关联已检索的知识？")
            sub_questions.append(f"最终答案应该包含哪些信息？")

        return sub_questions

    def _get_step_title(self, step_num: int) -> str:
        """获取步骤标题"""
        titles = {
            1: "📖 问题理解",
            2: "🔍 信息检索",
            3: "📊 分析推理",
            4: "✅ 综合结论"
        }
        return titles.get(step_num, f"步骤 {step_num}")

    def _classify_query_type(self, query: str) -> str:
        """分类问题类型"""
        for q_type, keywords in self._QUERY_TYPE_KEYWORDS:
            if any(kw in query for kw in keywords):
                return q_type
        return "综合类"

    def _extract_entities(self, query: str) -> List[str]:
        """基于规则提取关键实体"""
        entities = []

        for pattern in self._ENTITY_PATTERNS:
            for m in re.findall(pattern, query):
                if isinstance(m, tuple):
                    entities.extend([e for e in m if len(e) >= 2])
                elif len(m) >= 2:
                    entities.append(m)

        return list(set(entities))[:5]

    def _implicit_inference(self, sub_q: str, knowledge: str) -> str:
        """隐式推理 (模拟)"""
        if "类型" in sub_q:
            return "这是一个需要分析判断的问题类型"
        elif "实体" in sub_q or "关键" in sub_q:
            return "问题涉及的关键实体已识别"
        elif "关联" in sub_q or "知识" in sub_q:
            return "可以建立与知识的关联"
        elif "信息" in sub_q:
            return "答案框架已基本形成"
        return "推理进行中..."

    def _build_inference_prompt(self, query: str, knowledge: str) -> str:
        """构建推理提示"""
        return f"""
基于以下信息进行推理:
- 问题: {query}
- 已知知识: {knowledge[:500] if knowledge else '无'}

请分析:
1. 问题的核心是什么？
2. 知识如何支撑回答？
3. 推理的逻辑链是什么？
"""

    def _simulate_reasoning(self, query: str, knowledge: str,
                            evidence: List[str]) -> Dict[str, str]:
        """模拟推理过程（实际使用时由 LLM 完成）"""
        conclusion = f"综合查询'{query}'的相关信息，"

        if knowledge:
            conclusion += "基于检索到的外部知识，"
            if len(knowledge) > 200:
                conclusion += "可以得出较为完整的答案。"
            else:
                conclusion += "结合现有信息给出回答。"
        else:
            conclusion += "由于缺乏外部知识，采用模型自身知识库回答。"

        return {"conclusion": conclusion}

    def _synthesize_answer(self, query: str, conclusions: List[str],
                           knowledge: str) -> str:
        """综合生成最终答案（实际由 LLM 完成，此处为占位符）"""
        return f"【基于检索增强的思维链推理】关于'{query}'的综合回答"

    def _select_consistent_answer(self, answers: List[str]) -> str:
        """选择最一致的答案 (按长度接近均值投票)"""
        if not answers:
            return ""

        avg_length = sum(len(a) for a in answers) / len(answers)
        return min(answers, key=lambda a: abs(len(a) - avg_length))

    def _calculate_consistency(self, answers: List[str]) -> float:
        """计算答案一致性得分（字符级词汇重叠度）"""
        if len(answers) < 2:
            return 1.0

        scores = []
        for i, a1 in enumerate(answers):
            for a2 in answers[i+1:]:
                union = set(a1) | set(a2)
                if union:
                    scores.append(len(set(a1) & set(a2)) / len(union))

        return sum(scores) / len(scores) if scores else 1.0

    def build_cot_prompt(self, query: str, knowledge: str = "") -> str:
        """构建带有 CoT 的完整 prompt"""
        if self.mode == CoTMode.DIRECT:
            return self._build_direct_prompt(query, knowledge)
        elif self.mode == CoTMode.ZERO_SHOT:
            return self._build_zero_shot_prompt(query, knowledge)
        elif self.mode == CoTMode.FEW_SHOT:
            return self._build_few_shot_prompt(query, knowledge)
        elif self.mode == CoTMode.SELF_CONSISTENCY:
            return self._build_self_consistency_prompt(query, knowledge)

        return self._build_zero_shot_prompt(query, knowledge)

    def _build_direct_prompt(self, query: str, knowledge: str) -> str:
        """构建直接回答 prompt"""
        if knowledge:
            return f"===参考资料===\n{knowledge}\n\n===问题===\n{query}\n\n请直接给出简洁准确的回答："
        return query

    def _render_template(self, template: str, query: str, knowledge: str) -> str:
        """渲染 prompt 模板"""
        return template.format(
            knowledge=knowledge if knowledge else "（无参考资料）",
            query=query
        )

    def _build_zero_shot_prompt(self, query: str, knowledge: str) -> str:
        """构建 Zero-shot CoT prompt"""
        template = """===参考资料===
{knowledge}

===问题===
{query}

===要求===
请按以下步骤思考并回答：

1. **理解问题**：分析这个问题的核心是什么
2. **检索信息**：从参考资料中提取相关信息
3. **逻辑推理**：基于信息进行推理
4. **给出答案**：综合以上得出最终答案

请一步步思考："""
        return self._render_template(template, query, knowledge)

    def _build_few_shot_prompt(self, query: str, knowledge: str) -> str:
        """构建 Few-shot CoT prompt"""
        examples_text = "\n\n".join([
            f"示例 {i+1}:\n{ex.to_prompt()}"
            for i, ex in enumerate(self.examples[:2])
        ])

        template = """===示例学习===
{examples}

===参考资料===
{knowledge}

===当前问题===
{query}

请参考示例的推理方式，逐步思考并给出答案：

"""
        return template.format(
            examples=examples_text,
            knowledge=knowledge if knowledge else "（无参考资料）",
            query=query
        )

    def _build_self_consistency_prompt(self, query: str, knowledge: str) -> str:
        """构建 Self-Consistency prompt"""
        template = """===参考资料===
{knowledge}

===问题===
{query}

===要求===
请从多个角度思考这个问题，给出不同的推理路径和答案：

**路径A - 事实角度思考：**
（检索事实信息...）

**路径B - 概念角度思考：**
（分析核心概念...）

**路径C - 关系角度思考：**
（分析关联关系...）

请综合以上路径，给出最一致、最准确的答案："""
        return self._render_template(template, query, knowledge)


_default_reasoner: Optional[CoTReasoner] = None


def get_cot_reasoner(mode: CoTMode = CoTMode.ZERO_SHOT) -> CoTReasoner:
    """获取全局 CoT 推理器实例"""
    global _default_reasoner
    if _default_reasoner is None:
        _default_reasoner = CoTReasoner(mode=mode)
    return _default_reasoner


def reason_with_cot(query: str, knowledge: str = "",
                    mode: CoTMode = CoTMode.ZERO_SHOT,
                    depth: int = 1) -> Tuple[str, ReasoningChain]:
    """使用 CoT 进行推理，返回 (cot_prompt, reasoning_chain)"""
    reasoner = get_cot_reasoner(mode)
    chain = reasoner.reason(query, knowledge, depth)
    return reasoner.build_cot_prompt(query, knowledge), chain
