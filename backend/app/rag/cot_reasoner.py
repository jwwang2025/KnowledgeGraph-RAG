"""
CoT (Chain of Thought) 思维链推理模块

提供基于检索增强的思维链推理能力，包括：
1. Zero-shot CoT: 无示例的自动推理
2. Few-shot CoT: 基于示例的推理
3. Self-Consistency: 多路径推理一致性检查
4. 推理链可视化追踪
"""

import json
import re
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from enum import Enum
from config.settings import settings


class CoTMode(Enum):
    """思维链模式"""
    ZERO_SHOT = "zero_shot"           # Zero-shot CoT: "让我们一步步思考"
    FEW_SHOT = "few_shot"             # Few-shot CoT: 基于示例
    SELF_CONSISTENCY = "self_consistency"  # 自洽性推理
    DIRECT = "direct"                 # 直接回答 (无 CoT)


class ReasoningStep:
    """推理步骤"""
    step_num: int = 0
    title: str = ""
    content: str = ""
    evidence: List[str] = field(default_factory=list)  # 支撑证据
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
    
    # 推理步骤
    steps: List[ReasoningStep] = field(default_factory=list)
    
    # 中间结论
    intermediate_conclusions: List[str] = field(default_factory=list)
    
    # 最终答案
    final_answer: str = ""
    
    # 一致性检查 (仅 SELF_CONSISTENCY 模式)
    consistency_score: float = 0.0
    path_answers: List[str] = field(default_factory=list)
    
    # 元数据
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
        lines = [f"🔍 问题: {self.query}", ""]
        lines.append("📝 推理过程:")
        
        for i, step in enumerate(self.steps):
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
    reasoning_steps: List[str]  # 推理步骤列表
    answer: str
    
    def to_prompt(self) -> str:
        """转换为 prompt 格式"""
        lines = [f"问题: {self.question}"]
        for i, step in enumerate(self.reasoning_steps, 1):
            lines.append(f"推理 {i}: {step}")
        lines.append(f"答案: {self.answer}")
        return "\n".join(lines)


class CoTReasoner:
    """
    思维链推理器
    
    提供多种 CoT 模式来增强推理能力
    """
    
    # 默认的 Zero-shot CoT 后缀
    ZERO_SHOT_SUFFIX = "\n\n请逐步思考并给出答案："
    
    # 默认的 Few-shot 示例 (问答领域)
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
    
    def __init__(self, mode: CoTMode = CoTMode.ZERO_SHOT,
                 custom_examples: List[FewShotExample] = None,
                 enable_self_consistency: bool = False,
                 num_paths: int = 3):
        """
        初始化 CoT 推理器
        
        Args:
            mode: 思维链模式
            custom_examples: 自定义 Few-shot 示例
            enable_self_consistency: 启用自洽性检查
            num_paths: 自洽性检查的推理路径数
        """
        self.mode = mode
        self.examples = custom_examples or self.DEFAULT_EXAMPLES
        self.enable_self_consistency = enable_self_consistency
        self.num_paths = num_paths
        
        # 用于存储中间推理结果
        self._reasoning_cache: Dict[str, ReasoningChain] = {}
    
    def reason(self, query: str, knowledge_context: str,
               depth: int = 1) -> ReasoningChain:
        """
        执行思维链推理
        
        Args:
            query: 用户问题
            knowledge_context: 知识上下文
            depth: 推理深度 (1=简单, 2=深度)
            
        Returns:
            ReasoningChain: 推理链结果
        """
        chain = ReasoningChain(
            query=query,
            mode=self.mode,
            depth=depth
        )
        
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
        # 分解问题
        sub_questions = self._decompose_question(query, depth)
        
        for i, sub_q in enumerate(sub_questions, 1):
            step = ReasoningStep()
            step.step_num = i
            step.title = self._get_step_title(i, len(sub_questions))
            step.content = f"分析: {sub_q}"
            step.evidence = [knowledge[:300]] if knowledge else []
            
            # 模拟隐式推理
            step.conclusion = self._implicit_inference(sub_q, knowledge)
            
            chain.steps.append(step)
            chain.intermediate_conclusions.append(step.conclusion)
        
        # 生成最终答案
        chain.final_answer = self._synthesize_answer(query, chain.intermediate_conclusions, knowledge)
        chain.used_knowledge = [knowledge[:200]] if knowledge else []
        
        return chain
    
    def _few_shot_reasoning(self, query: str, knowledge: str,
                           chain: ReasoningChain, depth: int) -> ReasoningChain:
        """Few-shot CoT: 基于示例推理"""
        # 添加示例
        example_prompts = []
        for ex in self.examples[:2]:
            example_prompts.append(ex.to_prompt())
        
        # 问题分析步骤
        step1 = ReasoningStep()
        step1.step_num = 1
        step1.title = "📖 问题理解"
        step1.content = f"问题类型: {self._classify_query_type(query)}\n关键实体: {self._extract_entities(query)}"
        step1.evidence = example_prompts[:1]
        step1.conclusion = f"这是一个需要综合分析的问题"
        chain.steps.append(step1)
        
        # 知识匹配步骤
        if knowledge:
            step2 = ReasoningStep()
            step2.step_num = 2
            step2.title = "📚 知识检索"
            step2.content = f"从外部知识库检索到相关信息:\n{knowledge[:400]}"
            step2.evidence = [knowledge[:300]]
            step2.conclusion = f"找到 {len(knowledge)//100 + 1} 条相关信息"
            chain.steps.append(step2)
            chain.used_knowledge.append(knowledge[:200])
        
        # 推理分析步骤
        step3 = ReasoningStep()
        step3.step_num = 3
        step3.title = "🔍 推理分析"
        step3.content = self._build_inference_prompt(query, knowledge)
        step3.evidence = chain.used_knowledge
        
        # 模拟推理过程
        reasoning = self._simulate_reasoning(query, knowledge, chain.used_knowledge)
        step3.conclusion = reasoning.get("conclusion", "")
        chain.steps.append(step3)
        chain.intermediate_conclusions.append(step3.conclusion)
        
        # 综合结论步骤
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
        
        # 多路径推理
        for path_id in range(self.num_paths):
            path_chain = ReasoningChain(query=query, mode=self.mode, depth=depth)
            
            # 路径 1: 事实优先推理
            if path_id == 0:
                path_chain.steps.append(self._create_step(1, "事实检索", 
                    f"从知识图谱检索事实: {knowledge[:200]}", 
                    ["三元组信息"], "确认基础事实"))
            
            # 路径 2: 概念解释推理
            elif path_id == 1:
                path_chain.steps.append(self._create_step(1, "概念解析",
                    f"理解核心概念: {query}",
                    ["定义信息"], "明确概念内涵"))
            
            # 路径 3: 关系分析推理
            elif path_id == 2:
                path_chain.steps.append(self._create_step(1, "关系分析",
                    f"分析实体间关系: {knowledge[:200]}",
                    ["关系三元组"], "理清关联脉络"))
            
            # 添加推理步骤
            path_chain.steps.append(self._create_step(2, "逻辑推演",
                f"基于以上信息进行逻辑推导", [], "形成推理结论"))
            
            path_answer = self._synthesize_answer(query, [], knowledge)
            path_chain.final_answer = path_answer
            all_answers.append(path_answer)
        
        # 选择最一致的答案
        chain.path_answers = all_answers
        chain.final_answer = self._select_consistent_answer(all_answers)
        chain.consistency_score = self._calculate_consistency(all_answers)
        
        # 记录最终推理链
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
    
    # ==================== 辅助方法 ====================
    
    def _decompose_question(self, query: str, depth: int) -> List[str]:
        """分解问题为子问题"""
        sub_questions = []
        
        # 提取问题中的关键词
        keywords = self._extract_keywords(query)
        
        if depth >= 1:
            # 第一层: 识别问题类型
            q_type = self._classify_query_type(query)
            sub_questions.append(f"这个问题是什么类型？({q_type})")
        
        if depth >= 1:
            # 识别关键实体
            entities = self._extract_entities(query)
            if entities:
                sub_questions.append(f"问题涉及哪些关键实体？({', '.join(entities[:3])})")
        
        if depth >= 2:
            # 深度分析
            sub_questions.append(f"如何关联已检索的知识？")
            sub_questions.append(f"最终答案应该包含哪些信息？")
        
        return sub_questions
    
    def _get_step_title(self, step_num: int, total: int) -> str:
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
        if any(kw in query for kw in ["什么", "什么是", "定义"]):
            return "定义类"
        elif any(kw in query for kw in ["谁", "哪里", "什么时候"]):
            return "事实类"
        elif any(kw in query for kw in ["为什么", "原因"]):
            return "解释类"
        elif any(kw in query for kw in ["如何", "怎么", "方法"]):
            return "方法类"
        elif any(kw in query for kw in ["比较", "区别", "不同"]):
            return "比较类"
        return "综合类"
    
    def _extract_entities(self, query: str) -> List[str]:
        """提取关键实体"""
        # 简单基于规则的实体提取
        entities = []
        
        # 提取引号内容
        quoted = re.findall(r'[""]([^""]+)[""]', query)
        entities.extend(quoted)
        
        # 提取「」内容
        brackets = re.findall(r'[「』]([^「」]+)[「」]', query)
        entities.extend(brackets)
        
        # 提取常见实体类型
        patterns = [
            r'([\u4e00-\u9fa5]{2,})(是谁|是什么|在哪里)',
            r'([\u4e00-\u9fa5]{2,})和([\u4e00-\u9fa5]{2,})',
            r'的([\u4e00-\u9fa5]{2,})(是什么|有什么关系)',
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, query)
            for m in matches:
                if isinstance(m, tuple):
                    entities.extend([e for e in m if len(e) >= 2])
                elif len(m) >= 2:
                    entities.append(m)
        
        return list(set(entities))[:5]
    
    def _extract_keywords(self, query: str) -> List[str]:
        """提取关键词"""
        # 停用词
        stopwords = {'的', '是', '在', '了', '和', '与', '或', '有', '吗', '呢', '吧', '啊',
                    '什么', '如何', '怎么', '为什么', '哪个', '哪些'}
        
        words = [w for w in query if len(w) >= 2 and w not in stopwords]
        
        # 提取连续词组
        for length in [4, 3, 2]:
            for i in range(len(query) - length + 1):
                word = query[i:i+length]
                if word not in stopwords and not any(c in '，。！？、；：""''（）' for c in word):
                    words.append(word)
        
        return list(set(words))[:10]
    
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
        """模拟推理过程"""
        # 简化实现，实际使用时由 LLM 完成
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
        """综合生成最终答案"""
        # 这部分由 LLM 在实际调用时完成
        # 这里返回占位符
        return f"【基于检索增强的思维链推理】关于'{query}'的综合回答"
    
    def _select_consistent_answer(self, answers: List[str]) -> str:
        """选择最一致的答案 (投票)"""
        if not answers:
            return ""
        
        # 简单的相似度投票
        from collections import Counter
        
        # 统计回答长度作为简单聚类
        lengths = [len(a) for a in answers]
        avg_length = sum(lengths) / len(lengths)
        
        # 选择接近平均长度的答案
        best_answer = min(answers, key=lambda a: abs(len(a) - avg_length))
        
        return best_answer
    
    def _calculate_consistency(self, answers: List[str]) -> float:
        """计算答案一致性得分"""
        if len(answers) < 2:
            return 1.0
        
        # 简单的词汇重叠度
        scores = []
        for i, a1 in enumerate(answers):
            for a2 in answers[i+1:]:
                words1 = set(a1) & set(a2)
                words2 = set(a1) | set(a2)
                if words2:
                    score = len(words1) / len(words2)
                    scores.append(score)
        
        return sum(scores) / len(scores) if scores else 1.0
    
    # ==================== Prompt 构建 ====================
    
    def build_cot_prompt(self, query: str, knowledge: str = "") -> str:
        """
        构建带有 CoT 的完整 prompt
        
        Args:
            query: 用户问题
            knowledge: 知识上下文
            
        Returns:
            str: 完整的 CoT prompt
        """
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
        
        return template.format(
            knowledge=knowledge if knowledge else "（无参考资料）",
            query=query
        )
    
    def _build_few_shot_prompt(self, query: str, knowledge: str) -> str:
        """构建 Few-shot CoT prompt"""
        # 构建示例部分
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
        
        return template.format(
            knowledge=knowledge if knowledge else "（无参考资料）",
            query=query
        )


# ==================== 便捷函数 ====================

# 全局 CoT 推理器实例
_default_reasoner: Optional[CoTReasoner] = None


def get_cot_reasoner(mode: CoTMode = CoTMode.ZERO_SHOT) -> CoTReasoner:
    """获取 CoT 推理器实例"""
    global _default_reasoner
    if _default_reasoner is None:
        _default_reasoner = CoTReasoner(mode=mode)
    return _default_reasoner


def reason_with_cot(query: str, knowledge: str = "",
                   mode: CoTMode = CoTMode.ZERO_SHOT,
                   depth: int = 1) -> Tuple[str, ReasoningChain]:
    """
    使用 CoT 进行推理的便捷函数
    
    Args:
        query: 用户问题
        knowledge: 知识上下文
        mode: CoT 模式
        depth: 推理深度
        
    Returns:
        (cot_prompt, reasoning_chain): CoT prompt 和推理链
    """
    reasoner = get_cot_reasoner(mode)
    chain = reasoner.reason(query, knowledge, depth)
    prompt = reasoner.build_cot_prompt(query, knowledge)
    return prompt, chain