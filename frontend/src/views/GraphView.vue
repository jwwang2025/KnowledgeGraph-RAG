<template>
  <div class="graph-container">
    <div class="graph-controls">
      <div class="control-item">
        <label>节点过滤：</label>
        <select v-model="state.filterMode" @change="applyFilter">
          <option value="all">显示所有节点</option>
          <option value="top">仅显示重要节点（度数>平均值）</option>
          <option value="very-important">仅显示核心节点（度数>2倍平均值）</option>
        </select>
      </div>
      <div class="control-item">
        <label>节点数：{{ state.displayedNodes }}/{{ state.totalNodes }}</label>
      </div>
      <div class="control-item">
        <a-input-search
          v-model:value="state.searchText"
          placeholder="搜索实体..."
          style="width: 200px"
          @search="onSearch"
          @change="onSearchChange"
        />
      </div>
    </div>
    <div class="graph-content">
      <div id="graph-main" ref="chartRef"></div>
      <div id="graph-info" v-if="state.selectedNode">
        <div class="node-detail">
          <h3>{{ state.selectedNode.name }}</h3>
          <div class="node-type">
            <a-tag :color="getCategoryColor(state.selectedNode.category)">
              {{ state.selectedNode.category || '未分类' }}
            </a-tag>
          </div>
          <div class="node-stats">
            <div class="stat-item">
              <span class="stat-label">连接数</span>
              <span class="stat-value">{{ state.selectedNode.degree || 0 }}</span>
            </div>
            <div class="stat-item">
              <span class="stat-label">出现次数</span>
              <span class="stat-value">{{ state.selectedNode.lines?.length || 0 }}</span>
            </div>
          </div>
          <h4>关联描述</h4>
          <div class="node-sentences">
            <div
              v-for="(sent, idx) in state.colorfulSents"
              :key="idx"
              class="sent-item"
              v-html="sent"
            ></div>
          </div>
          <div class="node-actions">
            <a-button type="primary" size="small" @click="askAboutNode">
              询问此实体
            </a-button>
            <a-button size="small" @click="expandNode">
              展开邻接节点
            </a-button>
          </div>
        </div>
      </div>
      <div v-else class="graph-placeholder">
        <p>点击图谱中的节点查看详情</p>
        <p class="hint">节点大小表示连接数量，鼠标悬停查看名称</p>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted, onUnmounted, reactive } from 'vue'
import * as echarts from 'echarts'
import axios from 'axios'
import { useRouter } from 'vue-router'

const router = useRouter()
const chartRef = ref(null)
const state = reactive({
  graph: {},
  searchText: '',
  filterMode: 'all',
  displayedNodes: 0,
  totalNodes: 0,
  nodeDegree: {},
  avgDegree: 0,
  originalData: null,
  selectedNode: null,
  colorfulSents: []
})

let myChart

const getCategoryColor = (category) => {
  const colorMap = {
    '人物': 'blue',
    '地点': 'green',
    '组织': 'orange',
    '事件': 'red',
    '概念': 'purple',
    '术语': 'cyan',
    '物体': 'magenta'
  }
  return colorMap[category] || 'default'
}

const applyFilter = () => {
  if (!state.originalData) return

  let filteredNodes = []
  let filteredLinks = []
  const nodeMap = new Map()

  state.originalData.nodes.forEach((node, idx) => {
    const degree = state.nodeDegree[idx] || 0
    let shouldInclude = false

    if (state.filterMode === 'all') {
      shouldInclude = true
    } else if (state.filterMode === 'top') {
      shouldInclude = degree > state.avgDegree
    } else if (state.filterMode === 'very-important') {
      shouldInclude = degree > state.avgDegree * 2
    }

    if (shouldInclude) {
      nodeMap.set(idx, filteredNodes.length)
      filteredNodes.push({ ...node, originalIdx: idx })
    }
  })

  state.originalData.links.forEach(link => {
    if (nodeMap.has(link.source) && nodeMap.has(link.target)) {
      filteredLinks.push({
        ...link,
        source: nodeMap.get(link.source),
        target: nodeMap.get(link.target)
      })
    }
  })

  state.graph = {
    nodes: filteredNodes,
    links: filteredLinks,
    categories: state.originalData.categories,
    sents: state.originalData.sents
  }

  state.displayedNodes = filteredNodes.length
  updateChart()
}

const updateChart = () => {
  const webkitDep = state.graph

  const nodeDegree = {}
  webkitDep.links.forEach(function (link) {
    nodeDegree[link.source] = (nodeDegree[link.source] || 0) + 1
    nodeDegree[link.target] = (nodeDegree[link.target] || 0) + 1
  })

  const maxDegree = Math.max(...Object.values(nodeDegree), 1)

  webkitDep.nodes.forEach(function (node, idx) {
    const degree = nodeDegree[idx] || 0
    const normalizedDegree = maxDegree > 0 ? degree / maxDegree : 0
    node.symbolSize = 8 + normalizedDegree * 22

    const avgDegree = Object.values(nodeDegree).length > 0
      ? Object.values(nodeDegree).reduce((a, b) => a + b, 0) / webkitDep.nodes.length
      : 0
    node.label = {
      show: degree > avgDegree || node.symbolSize > 15,
      fontSize: 10,
      fontWeight: degree > avgDegree * 2 ? 'bold' : 'normal'
    }

    node.degree = degree
  })

  const option = {
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
      formatter: (params) => {
        if (params.dataType === 'node') {
          return `<strong>${params.data.name}</strong><br/>连接数: ${params.data.degree || 0}<br/>类别: ${params.data.category || '未分类'}`
        }
        return params.data.name
      }
    },
    series: [
      {
        type: 'graph',
        layout: 'force',
        animation: true,
        animationDuration: 1000,
        animationEasing: 'cubicOut',
        label: {
          position: 'right',
          formatter: '{b}',
          fontSize: 10,
          show: true
        },
        draggable: true,
        data: webkitDep.nodes.map(function (node, idx) {
          node.id = idx;
          return node;
        }),
        modularity: true,
        categories: webkitDep.categories,
        force: {
          edgeLength: 100,
          repulsion: 200,
          gravity: 0.02,
          friction: 0.6,
          layoutAnimation: true
        },
        lineStyle: {
          color: 'source',
          curveness: 0.1,
          width: 0.5,
          opacity: 0.6
        },
        edges: webkitDep.links,
        roam: true,
        focusNodeAdjacency: true,
        emphasis: {
          focus: 'adjacency',
          lineStyle: {
            width: 2,
            opacity: 1
          }
        }
      }
    ]
  }
  myChart.setOption(option)
}

const fetchWebkitDepData = () => {
  axios.get('/api/graph').then(response => response.data.data)
    .then(webkitDep => {
      state.originalData = webkitDep
      state.totalNodes = webkitDep.nodes.length

      const nodeDegree = {}
      webkitDep.links.forEach(function (link) {
        nodeDegree[link.source] = (nodeDegree[link.source] || 0) + 1
        nodeDegree[link.target] = (nodeDegree[link.target] || 0) + 1
      })

      state.nodeDegree = nodeDegree
      state.avgDegree = Object.values(nodeDegree).length > 0
        ? Object.values(nodeDegree).reduce((a, b) => a + b, 0) / webkitDep.nodes.length
        : 0

      myChart.hideLoading()

      applyFilter()
    })
}

const getNeighborNodes = (node) => {
  const nodes = []
  state.graph.links.forEach(function (link) {
    if (link.source === node.id || link.target === node.id) {
      nodes.push(state.graph.nodes[link.source])
      nodes.push(state.graph.nodes[link.target])
    }
  })

  nodes.forEach(function (item, index) {
    if (item.id === node.id) {
      nodes.splice(index, 1)
    }
  })

  return nodes
}

const colorfulSents = (node, nerborNodes, sents) => {
  const nerborNodeNames = nerborNodes.map((item) => item.name)
  const colorfulSents = sents.map((sent) => {
    let result = sent
    result = result.replace(
      new RegExp(node.name.replace(/[.*+?^${}()|[\]\\]/g, '\\$&'), 'g'),
      `<span style="color: #1890ff; font-weight: bold">${node.name}</span>`
    )
    nerborNodeNames.forEach((name) => {
      result = result.replace(
        new RegExp(name.replace(/[.*+?^${}()|[\]\\]/g, '\\$&'), 'g'),
        `<span style="color: #52c41a">${name}</span>`
      )
    })
    return result
  })
  return colorfulSents
}

const onSearch = (value) => {
  if (searchTimer) {
    clearTimeout(searchTimer)
    searchTimer = null
  }
  if (!value || !state.graph.nodes) return

  const targetNode = state.graph.nodes.find(
    node => node.name.toLowerCase().includes(value.toLowerCase())
  )

  if (targetNode) {
    clickNode({ dataType: 'node', data: targetNode })

    myChart.dispatchAction({
      type: 'focusNodeAdjacency',
      dataIndex: targetNode.id
    })
  }
}

let searchTimer = null

const onSearchChange = (e) => {
  state.searchText = e.target.value
  if (searchTimer) clearTimeout(searchTimer)
  searchTimer = setTimeout(() => {
    searchTimer = null
    onSearch(state.searchText)
  }, 300)
}

const clickNode = (param) => {
  if (param.dataType === 'node') {
    const node = param.data
    const sents = (node.lines || []).map((item) => state.graph.sents[item]).filter(Boolean)
    const nerborNodes = getNeighborNodes(node)

    state.selectedNode = {
      ...node,
      degree: node.degree || 0,
      lines: node.lines || []
    }
    state.colorfulSents = colorfulSents(node, nerborNodes, sents)
  }
}

const askAboutNode = () => {
  if (state.selectedNode) {
    router.push({
      path: '/chat',
      query: { q: state.selectedNode.name }
    })
  }
}

const expandNode = () => {
  if (state.selectedNode) {
    myChart.dispatchAction({
      type: 'focusNodeAdjacency',
      dataIndex: state.selectedNode.id
    })
  }
}

onMounted(() => {
  myChart = echarts.init(chartRef.value)
  myChart.showLoading()
  fetchWebkitDepData()
  myChart.on('click', clickNode)
})

onUnmounted(() => {
  if (myChart) {
    myChart.dispose()
    myChart = null
  }
  if (searchTimer) {
    clearTimeout(searchTimer)
    searchTimer = null
  }
})
</script>

<style lang="less" scoped>
.graph-container {
  display: flex;
  flex-direction: column;
  max-width: 100%;
  height: calc(100vh - 200px);
  gap: 10px;
}

.graph-controls {
  display: flex;
  flex-direction: row;
  align-items: center;
  gap: 20px;
  padding: 10px 20px;
  background: #fff;
  border-radius: 8px;
  box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);

  .control-item {
    display: flex;
    align-items: center;
    gap: 10px;

    label {
      font-size: 14px;
      color: #333;
      white-space: nowrap;
    }

    select {
      padding: 6px 12px;
      border: 1px solid #ddd;
      border-radius: 4px;
      font-size: 14px;
      background: #fff;
      cursor: pointer;
      transition: border-color 0.3s;

      &:focus {
        outline: none;
        border-color: #4a90e2;
      }
    }
  }
}

.graph-content {
  display: flex;
  flex: 1;
  gap: 10px;
  overflow: hidden;
}

#graph-main {
  flex: 1;
  background: #f5f5f5;
  border-radius: 8px;
  min-height: 400px;
}

#graph-info {
  width: 380px;
  background: #fff;
  border-radius: 8px;
  padding: 1.5rem;
  overflow-y: auto;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);

  &::-webkit-scrollbar {
    width: 6px;
  }

  &::-webkit-scrollbar-thumb {
    background: #ccc;
    border-radius: 3px;
  }
}

.node-detail {
  h3 {
    margin: 0 0 0.75rem 0;
    font-size: 1.25rem;
    color: #333;
  }

  .node-type {
    margin-bottom: 1rem;
  }

  .node-stats {
    display: flex;
    gap: 1.5rem;
    margin-bottom: 1.5rem;
    padding: 0.75rem;
    background: #f8f9fa;
    border-radius: 8px;

    .stat-item {
      display: flex;
      flex-direction: column;
      align-items: center;

      .stat-label {
        font-size: 0.75rem;
        color: #888;
        margin-bottom: 0.25rem;
      }

      .stat-value {
        font-size: 1.25rem;
        font-weight: 600;
        color: #1890ff;
      }
    }
  }

  h4 {
    margin: 0 0 0.75rem 0;
    font-size: 0.95rem;
    color: #333;
    border-bottom: 1px solid #e8e8e8;
    padding-bottom: 0.5rem;
  }

  .node-sentences {
    display: flex;
    flex-direction: column;
    gap: 0.5rem;
    max-height: 300px;
    overflow-y: auto;

    &::-webkit-scrollbar {
      width: 4px;
    }

    &::-webkit-scrollbar-thumb {
      background: #ddd;
      border-radius: 2px;
    }

    .sent-item {
      padding: 0.75rem;
      background: #f8f9fa;
      border-radius: 6px;
      font-size: 0.85rem;
      line-height: 1.6;
      color: #555;
      transition: background 0.2s;

      &:hover {
        background: #f0f0f0;
      }
    }
  }

  .node-actions {
    display: flex;
    gap: 0.75rem;
    margin-top: 1.5rem;

    button {
      flex: 1;
    }
  }
}

#graph-main,
#graph-info {
  display: flex;
  flex-direction: column;
  justify-content: start;
  align-items: center;
  background: #f5f5f5;
  border-radius: 8px;
}

#graph-info {
  width: 400px;
  padding: 2rem 1rem;
  overflow: scroll;
}

#graph-info {
  display: flex;
  flex-direction: column;
  justify-content: start;
  align-items: center;
  overflow: scroll;

  &::-webkit-scrollbar {
    display: none;
  }
}
</style>
