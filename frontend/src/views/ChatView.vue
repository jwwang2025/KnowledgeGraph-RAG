<template>
  <div class="chat-container">
    <div class="chat">
      <div class="chat-header">
        <h3>智能问答</h3>
        <div class="chat-mode-indicator">
          <span v-if="state.ragEnabled" class="mode-tag rag">RAG</span>
          <span v-if="state.cotEnabled" class="mode-tag cot">CoT</span>
          <span v-if="state.citationEnabled" class="mode-tag citation">引用</span>
        </div>
      </div>
      <div ref="chatBox" class="chat-box">
        <div
          v-for="message in state.messages"
          :key="message.id"
          class="message-box"
          :class="message.type"
        >
          <div v-if="message.citations && message.citations.length > 0" class="citations-bar">
            <span class="citations-label">参考来源：</span>
            <span v-for="(cite, idx) in message.citations" :key="idx" class="citation-item">
              [{{ cite.source }}]{{ cite.text?.slice(0, 20) }}...
            </span>
          </div>
          <img v-if="message.filetype === 'image'" :src="message.url" class="message-image" alt="">
          <div v-else class="message-content">
            <p v-if="message.cot_steps && message.cot_steps.length > 0" class="cot-reasoning">
              <span class="cot-label">推理过程：</span>
              <span v-for="(step, idx) in message.cot_steps" :key="idx" class="cot-step">{{ step }}</span>
            </p>
            <p style="white-space: pre-line" class="message-text">{{ message.text }}</p>
          </div>
          <div v-if="message.metadata" class="message-metadata">
            <a-collapse>
              <a-collapse-panel key="1" header="检索信息" :show-arrow="false" ghost>
                <div class="metadata-content">
                  <p><strong>问题类型：</strong>{{ message.metadata.question_type || '未知' }}</p>
                  <p><strong>置信度：</strong>{{ message.metadata.confidence || 0 }}%</p>
                  <p><strong>使用检索：</strong>{{ message.metadata.use_retrieval ? '是' : '否' }}</p>
                  <p><strong>知识源：</strong>{{ message.metadata.sources_used || 0 }} 个</p>
                  <p v-if="message.metadata.use_cot"><strong>推理模式：</strong>{{ message.metadata.cot_mode || 'direct' }}</p>
                </div>
              </a-collapse-panel>
            </a-collapse>
          </div>
        </div>
      </div>
      <div class="input-box">
        <a-button size="large" @click="clearChat" title="清空对话">
          <template #icon> <ClearOutlined /> </template>
        </a-button>
        <div class="input-controls">
          <a-switch v-model:checked="state.enableRAG" checked-children="RAG" un-checked-children="RAG" size="small" />
          <a-switch v-model:checked="state.enableCoT" checked-children="CoT" un-checked-children="CoT" size="small" />
        </div>
        <a-input
          type="text"
          class="user-input"
          v-model:value="state.inputText"
          @keydown.enter="sendMessage"
          placeholder="输入问题……"
        />
        <a-button size="large" type="primary" @click="sendMessage" :disabled="!state.inputText || state.isLoading">
          <template #icon> <SendOutlined /> </template>
        </a-button>
      </div>
    </div>
    <div class="info">
      <h1>{{ info.title }}</h1>

      <p class="description" v-if="info.description && typeof info.description === 'string'">{{ info.description }}</p>
      <div v-else-if="info.description && Array.isArray(info.description)">
        <p class="description" v-for="(desc, index) in info.description" :key="index">{{ desc }}</p>
      </div>

      <img v-if="info.image && typeof info.image === 'string'" :src="info.image" class="info-image" alt="">
      <div v-else-if="info.image && Array.isArray(info.image)">
        <img v-for="(img, index) in info.image" :key="index" :src="img" class="info-image" alt="">
      </div>

      <div v-if="info.ragStats" class="rag-stats">
        <h4>检索统计</h4>
        <p><span class="stat-label">问题类型：</span>{{ info.ragStats.question_type }}</p>
        <p><span class="stat-label">置信度：</span>{{ info.ragStats.confidence }}%</p>
        <p><span class="stat-label">使用知识源：</span>{{ info.ragStats.sources_used }} 个</p>
        <p><span class="stat-label">处理时间：</span>{{ info.ragStats.total_time }}</p>
        <div v-if="info.ragStats.stages && info.ragStats.stages.length > 0" class="stage-timeline">
          <span class="stage-label">处理阶段：</span>
          <span v-for="(stage, idx) in info.ragStats.stages" :key="idx" class="stage-item">{{ stage }}</span>
        </div>
      </div>

      <p v-show="hasGraphNodes"><b>关联图谱</b></p>
      <div id="lite_graph" v-show="hasGraphNodes"></div>
      <a-collapse v-model:activeKey="state.activeKey" v-if="hasGraphSents" accordion>
        <a-collapse-panel
          v-for="(sent, index) in info.graph.sents"
          :key="index"
          :header="'相关描述 ' + (index + 1)"
          :show-arrow="false"
          ghost
        >
          <p>{{ sent }}</p>
        </a-collapse-panel>
      </a-collapse>
    </div>
  </div>
</template>

<script setup>
import * as echarts from 'echarts';
import { reactive, ref, computed, onMounted, onUnmounted } from 'vue'
import { SendOutlined, ClearOutlined } from '@ant-design/icons-vue'

let myChart = null;
const chatBox = ref(null)
const state = reactive({
  history: [],
  messages: [],
  activeKey: [],
  inputText: '',
  enableRAG: true,
  enableCoT: true,
  isLoading: false,
  ragEnabled: false,
  cotEnabled: false,
  citationEnabled: false
})

const default_info = {
  title: 'KnowledgeGraph-RAG',
  description: [
  '依托RAG技术，融合知识图谱与文档库双源检索的智能问答系统，支持多轮对话与答案溯源，你可便捷实现：',
    '1. 图谱问答：输入问题，通过RAG技术调取知识图谱+文档双重证据，获取可溯源的精准答案',
    '2. 多轮筛选：在对话页通过实体、类别、类型等维度筛选，辅助RAG精准检索，快速定位专业知识',
    '3. 知识图谱可视化：在图谱页直观查看实体关联关系，支持缩放、平移、旋转，点击节点可查看详情',
    '4. 实体信息溯源：在图谱页右侧查看实体全场景出现记录，为RAG检索提供全面知识背景，助力深度理解',
  ],
  image: [],
  graph: null,
  ragStats: null
}

const info = reactive({
  ...default_info
})

const hasGraphNodes = computed(() => info.graph?.nodes?.length > 0)
const hasGraphSents = computed(() => info.graph?.sents?.length > 0)

const scrollToBottom = () => {
  setTimeout(() => {
    if (chatBox.value) {
      chatBox.value.scrollTop = chatBox.value.scrollHeight - chatBox.value.clientHeight
    }
  }, 10)
}

const appendMessage = (message, type, extra = {}) => {
  state.messages.push({
    id: state.messages.length + 1,
    type,
    text: message,
    ...extra
  })
  scrollToBottom()
}

const updateLastReceivedMessage = (message, id, extra = {}) => {
  const lastReceivedMessage = state.messages.find((message) => message.id === id)
  if (lastReceivedMessage) {
    Object.assign(lastReceivedMessage, { text: message, ...extra })
  } else {
    state.messages.push({
      id,
      type: 'received',
      text: message,
      ...extra
    })
  }
  scrollToBottom()
}

const sendMessage = () => {
  if (state.inputText.trim()) {
    appendMessage(state.inputText, 'sent')
    state.messages.push({
      id: state.messages.length + 1,
      type: 'received',
      text: '思考中...',
      isLoading: true
    })
    const cur_res_id = state.messages[state.messages.length - 1].id
    state.inputText = ''
    state.isLoading = true

    const requestData = {
      prompt: state.inputText.trim() || state.messages.filter(m => m.type === 'sent').pop()?.text,
      history: state.history
    }

    fetch('/api/chat', {
      method: 'POST',
      body: JSON.stringify(requestData),
      headers: {
        'Content-Type': 'application/json'
      }
    }).then((response) => {
      const reader = response.body.getReader()
      const decoder = new TextDecoder()
      let buffer = ''
      let graph
      let ragStats
      let citations
      let metadata
      let cotSteps

      const readChunk = () => {
        return reader.read().then(({ done, value }) => {
          if (done) {
            state.isLoading = false
            return
          }

          buffer += decoder.decode(value, { stream: true })
          const lines = buffer.trim().split('\n')
          buffer = lines.pop() || ''

          for (const line of lines) {
            try {
              const data = JSON.parse(line)

              state.ragEnabled = data.metadata?.rag?.use_retrieval || false
              state.cotEnabled = data.metadata?.cot?.mode !== 'direct' && data.metadata?.cot?.mode !== undefined
              state.citationEnabled = !!(data.citations && data.citations.length > 0)

              if (data.metadata?.rag) {
                ragStats = data.metadata.rag
                info.ragStats = {
                  question_type: ragStats.question_type || '未知',
                  confidence: Math.round((ragStats.confidence || 0) * 100),
                  sources_used: ragStats.sources_used || 0,
                  total_time: ragStats.total_time || '0s',
                  stages: ragStats.stages || [],
                  use_cot: ragStats.use_cot || false,
                  cot_mode: ragStats.cot_mode || 'direct'
                }
              }

              if (data.citations) {
                citations = data.citations
              }

              if (data.metadata?.cot?.steps) {
                cotSteps = Array(data.metadata.cot.steps).fill('')
              }

              info.image = data.image
              info.graph = data.graph
              info.title = data.wiki?.title
              info.description = data.wiki?.summary

              if (info.graph && info.graph.nodes) {
                myChart.setOption(graphOption(info.graph))
              }

              if (data.updates?.response) {
                updateLastReceivedMessage(
                  data.updates.response,
                  cur_res_id,
                  {
                    citations,
                    metadata: data.metadata?.rag,
                    cot_steps: cotSteps
                  }
                )
              }

              if (data.history) {
                state.history = data.history
              }

            } catch (e) {
              console.error('Parse error:', e)
            }
          }

          return readChunk()
        })
      }
      return readChunk()
    }).catch(err => {
      state.isLoading = false
      console.error('Request error:', err)
      updateLastReceivedMessage('请求失败，请重试', cur_res_id)
    })
  }
}

const graphOption = (graph) => {
  graph.nodes.forEach(node => {
    node.symbolSize = 5;
    node.label = {
      show: true
    }
  });
  let option = {
    tooltip: {
      show: true,
      showContent: true,
      trigger: 'item',
      triggerOn: 'mousemove',
      alwaysShowContent: false,
      showDelay: 0,
      hideDelay: 200,
      enterable: false,
      position: 'right',
      confine: false,
      formatter: (x) => x.data.name
    },
    series: [
      {
        type: 'graph',
        draggable: true,
        layout: 'force',
        data: graph.nodes.map(function (node, idx) {
          node.id = idx;
          return node;
        }),
        links: graph.links,
        categories: graph.categories,
        roam: true,
        label: {
          position: 'right'
        },
        force: {
          repulsion: 100
        },
        lineStyle: {
          color: 'source',
          curveness: 0.1
        },
      }
    ]
  };

  return option
}


const sendDeafultMessage = () => {
  setTimeout(() => {
    appendMessage('你好！我是 KnowledgeGraph-RAG 智能问答助手，基于 Adaptive-RAG + Self-RAG + CoT 技术。\n\n我可以帮你：\n- 通过知识图谱检索答案\n- 追踪答案的引用来源\n- 展示推理过程\n\n请开始提问吧！', 'received')
  }, 500);
}

const clearChat = () => {
  state.messages = []
  state.history = []
  info.title = default_info.title
  info.description = default_info.description
  info.image = default_info.image
  info.graph = default_info.graph
  info.ragStats = null
  sendDeafultMessage()
}

onMounted(() => {
  sendDeafultMessage()
  myChart = echarts.init(document.getElementById('lite_graph'));
})

onUnmounted(() => {
  if (myChart) {
    myChart.dispose()
    myChart = null
  }
})
</script>

<style lang="less" scoped>
.chat-container {
  display: flex;
  gap: 1.5rem;
}

.chat {
  display: flex;
  width: 100%;
  max-width: 800px;
  flex-grow: 1;
  margin: 0 auto;
  flex-direction: column;
  height: calc(100vh - 135px);
  background: #f5f5f5;
  border-radius: 8px;
  box-shadow: 0px 0.3px 0.9px rgba(0, 0, 0, 0.12), 0px 0.6px 2.3px rgba(0, 0, 0, 0.1),
    0px 1px 5px rgba(0, 0, 0, 0.08);
}

.chat-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 0.75rem 1rem;
  border-bottom: 1px solid #e8e8e8;
  background: #fff;

  h3 {
    margin: 0;
    font-size: 1rem;
    color: #333;
  }

  .chat-mode-indicator {
    display: flex;
    gap: 0.5rem;

    .mode-tag {
      padding: 0.2rem 0.5rem;
      border-radius: 4px;
      font-size: 0.7rem;
      font-weight: 600;

      &.rag {
        background: #e6f7ff;
        color: #1890ff;
      }

      &.cot {
        background: #fff7e6;
        color: #fa8c16;
      }

      &.citation {
        background: #f6ffed;
        color: #52c41a;
      }
    }
  }
}

.chat-box {
  flex: 1;
  overflow-y: auto;
  padding: 1rem;
  display: flex;
  flex-direction: column;
  scroll-behavior: smooth;

  &::-webkit-scrollbar {
    width: 0.4rem;
  }

  &::-webkit-scrollbar-thumb {
    background: #ccc;
    border-radius: 4px;
  }
}

.message-box {
  width: fit-content;
  display: inline-block;
  padding: 0.5rem;
  border-radius: 0.5rem;
  margin: 0.5rem 0;
  box-sizing: border-box;
  padding: 10px 16px;
  user-select: text;
  word-break: break-word;
  font-size: 14px;
  line-height: 20px;
  font-weight: 400;
  box-shadow: 0px 0.3px 0.9px rgba(0, 0, 0, 0.12), 0px 1.6px 3.6px rgba(0, 0, 0, 0.16);
  max-width: 80%;

  &.sent {
    color: white;
    background: linear-gradient(90deg, #40788c 10.79%, #005f77 87.08%);
    align-self: flex-end;
  }

  &.received {
    color: #111111;
    background-color: #ffffff;
    text-align: left;
    align-self: flex-start;
  }
}

.message-content {
  display: flex;
  flex-direction: column;
  gap: 0.5rem;
}

.citations-bar {
  display: flex;
  flex-wrap: wrap;
  gap: 0.25rem;
  padding: 0.5rem;
  margin-bottom: 0.5rem;
  background: #f8f8f8;
  border-radius: 6px;
  font-size: 0.75rem;

  .citations-label {
    color: #666;
    font-weight: 500;
  }

  .citation-item {
    color: #1890ff;
    background: #e6f7ff;
    padding: 0.1rem 0.4rem;
    border-radius: 3px;
    max-width: 150px;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }
}

.cot-reasoning {
  background: #fffbe6;
  border-left: 3px solid #faad14;
  padding: 0.5rem;
  margin: 0.25rem 0;
  border-radius: 0 4px 4px 0;
  font-size: 0.85rem;

  .cot-label {
    color: #d48806;
    font-weight: 600;
  }

  .cot-step {
    display: block;
    margin-top: 0.25rem;
    color: #555;
  }
}

.message-metadata {
  margin-top: 0.5rem;
  border-top: 1px dashed #e8e8e8;
  padding-top: 0.5rem;

  .metadata-content {
    font-size: 0.8rem;
    color: #666;

    p {
      margin: 0.25rem 0;
    }
  }
}

p.message-text {
  word-wrap: break-word;
  margin-bottom: 0;
}

img.message-image {
  max-width: 300px;
  max-height: 50vh;
  object-fit: contain;
}

.input-box {
  display: flex;
  align-items: center;
  padding: 1rem;
  border-top: 1px solid #ccc;
  gap: 0.5rem;

  .input-controls {
    display: flex;
    gap: 0.5rem;
    margin-right: 0.5rem;
  }
}

input.user-input {
  flex: 1;
  height: 40px;
  padding: 0.5rem 1rem;
  background-color: #fff;
  border: 1px solid #d9d9d9;
  border-radius: 8px;
  font-size: 16px;
  color: #111111;
  transition: border-color 0.3s;

  &:focus {
    border-color: #1890ff;
    outline: none;
    box-shadow: 0 0 0 2px rgba(24, 144, 255, 0.2);
  }
}

.ant-btn-icon-only {
  font-size: 16px;
  border-radius: 8px;
  cursor: pointer;
}

div.info {
  width: 400px;
  min-width: 400px;
  height: calc(100vh - 135px);
  overflow-y: auto;
  flex-grow: 0;
  scroll-behavior: smooth;

  &::-webkit-scrollbar {
    width: 0.4rem;
  }

  &::-webkit-scrollbar-thumb {
    background: #ccc;
    border-radius: 4px;
  }

  & > h1 {
    font-size: 1.5rem;
    margin: 0.5rem 0;
  }

  p.description {
    font-size: 1rem;
    margin: 0;
    margin-bottom: 20px;
  }

  img {
    width: 100%;
    height: fit-content;
    object-fit: contain;
    border-radius: 8px;
    overflow: hidden;
    margin-bottom: 0.5rem;
  }

  #lite_graph {
    width: 400px;
    height: 300px;
    background: #f5f5f5;
    border-radius: 8px;
    margin-bottom: 1rem;
    box-shadow: 0px 0.3px 0.9px rgba(0, 0, 0, 0.12), 0px 0.6px 2.3px rgba(0, 0, 0, 0.1),
      0px 1px 5px rgba(0, 0, 0, 0.08);
  }
}

.rag-stats {
  background: #f8f9fa;
  border-radius: 8px;
  padding: 1rem;
  margin-bottom: 1rem;
  box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);

  h4 {
    margin: 0 0 0.75rem 0;
    font-size: 1rem;
    color: #333;
    border-bottom: 1px solid #e8e8e8;
    padding-bottom: 0.5rem;
  }

  p {
    margin: 0.35rem 0;
    font-size: 0.85rem;
    color: #555;
  }

  .stat-label {
    color: #888;
  }

  .stage-timeline {
    display: flex;
    flex-wrap: wrap;
    gap: 0.25rem;
    margin-top: 0.5rem;

    .stage-label {
      color: #888;
      font-size: 0.85rem;
    }

    .stage-item {
      background: #e6f7ff;
      color: #1890ff;
      padding: 0.15rem 0.4rem;
      border-radius: 3px;
      font-size: 0.7rem;
    }
  }
}
</style>
