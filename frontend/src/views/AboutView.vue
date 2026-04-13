<template>
  <div class="about">
    <h1>KnowledgeGraph-RAG</h1>
    <h2>项目简介</h2>
    <p>设计和实现了一个检索增强生成(RAG)对话系统，创新性地融合知识图谱与向量数据库进行双源检索，结合自适应路由策略和思维链推理，为用户提供精准、可溯源的专业问答服务。系统能够自动分析用户问题类型，智能选择最优</p>
    <p>检索策略，并对检索结果进行质量评估和多轮优化重排序，显著提升了问答的准确性和可靠性。</p>
    <p>本项目提出一种 "结构化知识图谱+非结构化文档库" 双检索驱动的检索增强生成（RAG）对话系统，核心目标是解决大模型在事实性问答中存在的 <strong>幻觉</strong> 问题，同时提升回答的可追溯性与上下文相关性。</p>
    <p>系统以知识图谱（结构化知识）和文档库（非结构化知识）为双重外部知识源，通过 RAG 技术将检索到的精准事实证据动态注入生成模型（ChatGLM-6B），最终实现 <strong>有依据、可验证、高准确</strong> 的智能对话服务。</p>

    <h2>技术特性</h2>
    <ul>
      <li><strong>UIE 抽取</strong>：高效完成实体、关系、属性的自动化抽取</li>
      <li><strong>SPN4RE</strong>：提升关系三元组抽取的精度与效率</li>
      <li><strong>双源 RAG 检索增强</strong>：融合知识图谱结构化检索与文档库非结构化检索</li>
      <li><strong>Adaptive-RAG 智能路由</strong>：根据问题类型自适应选择检索策略和知识源</li>
      <li><strong>Self-RAG 结果评估</strong>：评估检索结果相关性，智能过滤低质量内容</li>
      <li><strong>CoT 思维链推理</strong>：支持 Zero-shot / Few-shot / Self-Consistency 多种推理模式</li>
      <li><strong>引用溯源（Citations）</strong>：确保回答中关键细节与检索来源可追溯验证</li>
    </ul>

    <h2>核心功能</h2>
    <p><strong>智能问答</strong>：在对话页面进行多轮追问，系统自动分析问题类型，选择最优检索策略，结合知识图谱与文档库生成有依据的回答。</p>
    <p><strong>知识图谱可视化</strong>：在图谱页面浏览实体关系网络，点击节点查看详情，支持关系筛选与路径探索。</p>
    <p><strong>检索增强生成</strong>：基于 Adaptive-RAG + Self-RAG + CoT 三层架构，实现智能路由、多源检索、结果评估与思维链推理。</p>
    <p><strong>引用溯源</strong>：每条回答都附带引用来源，点击即可查看原始证据，确保回答可验证。</p>

    <h2>技术栈</h2>
    <ul>
      <li><strong>前端</strong>：Vue 3 + Ant Design Vue + ECharts + D3.js + Vite</li>
      <li><strong>后端</strong>：Flask + Python 3.x</li>
      <li><strong>模型</strong>：ChatGLM-6B（生成）、UIE（信息抽取）、BERT（嵌入）</li>
      <li><strong>向量库</strong>：ChromaDB + Sentence Transformers</li>
    </ul>

    <h2>使用场景</h2>
    <p><strong>专业领域问答</strong>：适用于需要准确事实依据的专业问题解答。</p>
    <p><strong>知识探索</strong>：通过图谱可视化了解实体关系与知识网络结构。</p>
    <p><strong>文档检索</strong>：在海量文档中快速定位相关信息并获取精准答案。</p>

    <h2>系统架构</h2>
    <p>系统采用 <strong>Adaptive-RAG + Self-RAG + CoT</strong> 三层架构：</p>
    <ul>
      <li><strong>Adaptive-RAG</strong>：根据问题类型自适应选择检索策略（问题路由 → 检索决策）</li>
      <li><strong>Self-RAG</strong>：评估检索结果相关性，通过 RRF 融合与 Cohere 语义重排序优化结果</li>
      <li><strong>CoT</strong>：支持多种思维链推理模式，提升复杂问题的回答质量</li>
    </ul>
  </div>
</template>

<style lang="less" scoped>
.about {
  display: flex;
  flex-direction: column;
  align-items: flex-start;
  color: #333;
  width: 80%;
  max-width: 800px;
  min-height: calc(100vh - 200px);
  margin: 0 auto;
  margin-top: 40px;
  font-family: 'HarmonyOS Sans SC', 'Noto Sans SC', 'Source Han Sans SC', 'PingFang SC', 'Microsoft YaHei', 'WenQuanYi Micro Hei', 'WenQuanYi Zen Hei', 'ST Heiti', 'SimHei', 'WenQuanYi Zen Hei Sharp', 'Droid Sans Fallback', 'Droi';
}

h1 {
  width: 100%;
  text-align: center;
  font-weight: 700;
  margin: 1rem 0;
  color: #005f77;
}

h2 {
  width: 100%;
  text-align: left;
  font-size: 1.2rem;
  line-height: 1.2rem;
  margin: 1em 0;
  border-left: 5px solid #005f77;
  padding-left: 0.8em;
}

p {
  text-align: left;
  font-size: 1rem;
  line-height: 1.8em;
  margin: 0.5em 0;
}

ul {
  text-align: left;
  font-size: 1rem;
  line-height: 1.8em;
  margin: 0.5em 0;
  padding-left: 2em;
  
  li {
    margin: 0.3em 0;
  }
}

footer {
  margin-top: auto;
  text-align: center;
  font-size: 0.8em;
  color: #999;

  a {
    color: #999;
  }
}
</style>