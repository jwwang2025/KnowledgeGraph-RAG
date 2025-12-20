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
    </div>
    <div id="graph-main" ref="chartRef"></div>
    <!-- <div id="graph-info">
      <div class="search-area">
        <input type="text" v-model="state.searchText" placeholder="输入关键词搜索" />
      </div>
      <div class="node-info">
        <div class="node-card" v-for="(sent, idx) in state.nodeInfo" :key="idx">
          <div class="node-sent" v-html="sent"></div>
        </div>
      </div>
    </div> -->
  </div>
</template>

<script setup>
import { ref, onMounted, reactive } from 'vue'
import * as echarts from 'echarts'
import axios from 'axios'

const chartRef = ref(null)
const state = reactive({
  graph: {},
  searchText: '',
  showInfo: true,
  nodeInfo: [],
  filterMode: 'all',
  displayedNodes: 0,
  totalNodes: 0,
  nodeDegree: {},
  avgDegree: 0,
  originalData: null
})

let myChart;

const applyFilter = () => {
  if (!state.originalData) return
  
  let filteredNodes = []
  let filteredLinks = []
  const nodeMap = new Map()
  
  // 根据过滤模式筛选节点
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
  
  // 筛选边：只保留两个端点都在过滤后节点集中的边
  state.originalData.links.forEach(link => {
    if (nodeMap.has(link.source) && nodeMap.has(link.target)) {
      filteredLinks.push({
        ...link,
        source: nodeMap.get(link.source),
        target: nodeMap.get(link.target)
      })
    }
  })
  
  // 更新显示数据
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
  
  // 重新计算度数（基于过滤后的数据）
  const nodeDegree = {}
  webkitDep.links.forEach(function (link) {
    nodeDegree[link.source] = (nodeDegree[link.source] || 0) + 1
    nodeDegree[link.target] = (nodeDegree[link.target] || 0) + 1
  })
  
  // 找到最大度数，用于归一化
  const maxDegree = Math.max(...Object.values(nodeDegree), 1)
  
  // 处理节点：调整大小和标签显示
  webkitDep.nodes.forEach(function (node, idx) {
    const degree = nodeDegree[idx] || 0
    // 根据度数计算节点大小，范围在 8-30 之间
    const normalizedDegree = maxDegree > 0 ? degree / maxDegree : 0
    node.symbolSize = 8 + normalizedDegree * 22
    
    // 根据度数决定是否显示标签
    const avgDegree = Object.values(nodeDegree).length > 0 
      ? Object.values(nodeDegree).reduce((a, b) => a + b, 0) / webkitDep.nodes.length 
      : 0
    node.label = {
      show: degree > avgDegree || node.symbolSize > 15,
      fontSize: 10,
      fontWeight: degree > avgDegree * 2 ? 'bold' : 'normal'
    }
    
    // 保存度数信息用于后续使用
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
          return `${params.data.name}<br/>连接数: ${params.data.degree || 0}`
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
        modularity: true, // 开启社区划分
        categories: webkitDep.categories,
        force: {
          // 优化力导向图参数
          edgeLength: 100,  // 增加边长度，让节点分散更开
          repulsion: 200,   // 大幅增加排斥力，防止节点聚集
          gravity: 0.02,    // 降低重力，减少向中心聚集的趋势
          friction: 0.6,    // 添加摩擦力，使布局更稳定
          layoutAnimation: true  // 启用布局动画
        },
        lineStyle: {
          color: 'source',
          curveness: 0.1,
          width: 0.5,  // 减小边宽度，减少视觉混乱
          opacity: 0.6  // 降低边的不透明度
        },
        edges: webkitDep.links,
        roam: true, // 开启鼠标缩放和平移漫游
        focusNodeAdjacency: true, // 高亮显示鼠标移入节点的邻接节点
        emphasis: {
          // 鼠标悬停时的样式
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
      // 保存原始数据
      state.originalData = webkitDep
      state.totalNodes = webkitDep.nodes.length
      
      // 计算每个节点的度数（连接数）
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
      
      // 应用初始过滤
      applyFilter()
    })
}

const getNeighborNodes = (node) => {
  const nodes = []
  // 遍历所有的边，找到与当前节点相连的节点
  state.graph.links.forEach(function (link) {
    if (link.source === node.id || link.target === node.id) {
      nodes.push(state.graph.nodes[link.source])
      nodes.push(state.graph.nodes[link.target])
    }
  })

  // 去除当前节点
  nodes.forEach(function (item, index) {
    if (item.id === node.id) {
      nodes.splice(index, 1)
    }
  })

  return nodes
}

const colorfulSents = (node, nerborNodes, sents) => {
  const nerborNodeNames = nerborNodes.map((item) => item.name)
  console.log(nerborNodeNames)
  const colorfulSents = sents.map((sent) => {
    sent = sent.replace(node.name, `<span style="color: #47c640">${node.name}</span>`)
    nerborNodeNames.forEach((name) => {
      sent = sent.replace(name, `<span style="color: #df2024">${name}</span>`)
    })
    return sent
  })
  return colorfulSents
}

const clickNode = (param) => {
  console.log('点击了', param)

  if (param.dataType === 'node') {
    state.showInfo = true
    const sents = param.data.lines.map((item) => state.graph.sents[item])
    const nerborNodes = getNeighborNodes(param.data)
    state.nodeInfo = colorfulSents(param.data, nerborNodes, sents)
  }
}

onMounted(() => {
  myChart = echarts.init(chartRef.value)
  myChart.showLoading()
  fetchWebkitDepData()
  myChart.on('click', clickNode)
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

#graph-main,
#graph-info {
  display: flex;
  flex-direction: column;
  justify-content: start;
  align-items: center;
  height: 100%;
  background: #f5f5f5;
  border-radius: 8px;
}

#graph-main {
  width: 100%;
}

#graph-info {
  width: 400px;
  padding: 2rem 1rem;
  overflow: scroll;

  .search-area {
    // 优化 input 和 button 的样式
    display: flex;
    flex-direction: row;
    justify-content: center;
    align-items: center;
    width: 100%;
    height: 40px;
    margin-bottom: 1rem;

    input {
      flex: 1;
      width: 100%;
      padding: 0.5rem 1rem;
      background-color: #fff;
      border: none;
      border-radius: 8px;
      // box-shadow: 0 0 0 1px rgba(0, 0, 0, 0.1), 0 1px 2px rgba(0, 0, 0, 0.1);
      font-size: 0.8rem;
      margin: 1rem 1rem;
      color: #111111;
      line-height: 22px;
      font-variation-settings: 'wght' 400, 'opsz' 10.5;
      transition: all 0.3s;
    }

    input:focus {
      outline: 2px solid #999;
    }

    // place holder
    input::-webkit-input-placeholder {
      color: #999999;
    }
  }
}

#graph-info,
.node-info {
  display: flex;
  flex-direction: column;
  justify-content: start;
  align-items: center;
  overflow: scroll;

  // 隐藏滚动条
  &::-webkit-scrollbar {
    display: none;
  }
}

.node-info {
  display: flex;
  flex-direction: column;
  justify-content: start;
  align-items: center;
  width: 100%;
  height: 100%;
  overflow: scroll;
}

#graph-info .node-sent {
  margin: 1rem 0;
  padding: 1rem;
  background: #fff;
  border-radius: 8px;
  box-shadow: 0 0 0 1px rgba(0, 0, 0, 0.05);
  font-size: 0.8rem;
  color: #111111;
  line-height: 22px;
  font-variation-settings: 'wght' 400, 'opsz' 10.5;
  transition: all 0.3s;
}
</style>
