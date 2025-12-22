<template>
  <div class="about">
    <img class="logo" src="/teamnum.png" alt="">
    <h2>项目简介</h2>
    <p>本项目是一个面向专业领域的本地化知识图谱检索与问答系统（RAG），将结构化知识与大模型生成能力结合，便于快速搭建领域问答与可视化应用。</p>

    <h2>核心功能</h2>
    <p>知识图谱检索：基于预处理与抽取的数据，支持实体、关系与属性的检索与展示。</p>
    <p>检索增强生成（RAG）：将检索结果作为上下文，结合 ChatGLM 生成更有依据的回答。</p>
    <p>统一信息抽取（UIE）：用于从文本与图片中抽取实体与关系，辅助构建与补全图谱。</p>
    <p>多模态与可视化：支持文本/图片混合展示，并使用 ECharts 可视化知识图谱与关系网络。</p>

    <h2>使用场景</h2>
    <p>交互式问答：在对话页面进行多轮追问与筛选，以逐步聚焦用户意图。</p>
    <p>知识探索：在图谱页面浏览实体关系，点击节点查看详情并高亮关联内容。</p>
    <p>知识导入与补全：通过预处理与抽取模块，将新的领域资料加入知识库以扩充覆盖率。</p>

    <h2>技术栈</h2>
    <p>前端：Vue 3 + Vite，负责交互与可视化展示。</p>
    <p>后端：Flask 提供检索、RAG 接口与模型服务适配。</p>
    <p>模型与数据：集成本地 ChatGLM、BERT 等模型，配合自定义预处理与构建脚本。</p>

    <h2>快速上手</h2>
    <p>项目可在本地部署与测试，参见根目录 README 以及 `chat-kg/0_README.md` 中的启动说明。</p>
    <p>如需扩展领域数据，请使用 `modules/prepare` 中的预处理与构建脚本进行导入与标注。</p>
  </div>
</template>

<style lang="less" scoped>
.about {
  display: flex;
  flex-direction: column;
  align-items: center;
  color: #333;
  width: 80%;
  max-width: 800px;
  min-height: calc(100vh - 200px);
  margin: 0 auto;
  margin-top: 40px;
  // text-align: center;
  font-family: 'HarmonyOS Sans SC', 'Noto Sans SC', 'Source Han Sans SC', 'PingFang SC', 'Microsoft YaHei', 'WenQuanYi Micro Hei', 'WenQuanYi Zen Hei', 'ST Heiti', 'SimHei', 'WenQuanYi Zen Hei Sharp', 'Droid Sans Fallback', 'Droi';
}

img.logo {
  width: 300px;
  // height: 100px;
  margin: 2rem auto;
}

img {
  max-width: 100%;
  margin: 1rem auto;
  object-fit: contain;
  border-radius: 8px;
}

h2 {
  width: 100%;
  font-size: 1.2rem;
  line-height: 1.2rem;
  margin: 1em 0;
  border-left: 5px solid #005f77;
  padding-left: 0.8em;
}

p {
  font-size: 1rem;
  line-height: 1.8em;
  margin: 0.5em 0;
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
