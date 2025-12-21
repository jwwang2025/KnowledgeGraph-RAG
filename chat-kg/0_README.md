# chat-kg 目录说明

``` 
chat-kg/
├── src/                  # 源代码
│   ├── components/       # 复用组件（聊天框、消息列表、图谱画布等）
│   ├── pages/            # 页面级组件（主聊天页、图谱查看页）
│   ├── api/              # 后端 API 封装（/api/chat, /api/graph）
│   └── assets/           # 样式与静态资源
├── public/               # 静态文件
├── vite.config.js        # Vite 配置（端口、代理、base 等）
└── package.json          # 依赖与 npm 脚本（dev, server, build, preview）
```

本文件仅列出 `chat-kg` 的目录结构，详细使用与联调说明请参见仓库根 `README.md`。
本目录为项目的前端聊天界面，实现与后端知识图谱检索增强（RAG）服务的交互与可视化。该说明参考仓库根 `README.md` 的项目结构与运行流程，补充本子模块的目录说明、启动、联调与常见问题，帮助快速上手。

目标
- 提供基于 Vue 3 的聊天界面：对话输入、检索结果展示、知识图谱可视化（图谱节点/关系展示）和多轮会话管理。

目录结构（示例）
- `src/`：前端源代码
  - `components/`：复用组件（聊天框、消息列表、输入框、图谱画布等）
  - `pages/`：页面级组件（主聊天页面、图谱查看页）
  - `api/`：与后端交互的封装（封装 `/api/chat`、`/api/graph` 等）
  - `assets/`：样式与静态资源
- `public/`：静态文件
- `vite.config.js`：Vite 配置（开发端口、代理、资源 base 等）
- `package.json`：依赖与 npm 脚本（如 `dev`、`server`、`build`、`preview`）

运行与联调（快速指南）

1) 安装依赖

```bash
# 在仓库根或进入本目录执行
cd chat-kg
npm install
```

2) 本地开发（带热重载）

```bash
# 启动 dev
npm run dev
# 或若项目提供 server 脚本以允许局域网访问：
npm run server
```

3) 修改开发端口或代理（可选）

- 在 `vite.config.js` 中修改 `server.port`（避免端口冲突）。
- 若需将前端请求代理到本地后端，配置 `server.proxy`，或通过环境变量（`.env`）指定后端基地址，例如：

```env
VITE_API_BASE=http://localhost:5000
```

或在 `vite.config.js` 中添加代理：

```js
// vite.config.js（示例）
export default defineConfig({
  server: {
    port: 5173,
    proxy: {
      '/api': {
        target: 'http://localhost:5000',
        changeOrigin: true,
        secure: false
      }
    }
  }
})
```

4) 与后端配合

- 后端默认地址：`http://localhost:5000`（参见仓库根 README 的后端启动说明）。
- 关键 API：
  - `/api/chat`：会话/对话接口
  - `/api/graph`：知识图谱查询与可视化数据接口
- 开发时建议先启动后端并确保 API 可用，再启动前端以避免跨域或 404 错误。

打包与部署

```bash
# 生成生产构建
npm run build
# 可选：本地预览构建产物
npm run preview
```

生成的静态文件位于 `dist/`（默认），可部署到任意静态托管服务（如 GitHub Pages、Nginx、Netlify 等）。

常见问题与排查
- 前端无法访问后端：确认后端已启动、检查 `VITE_API_BASE` 或 `proxy` 配置、检查防火墙设置。
- 端口冲突：修改 `vite.config.js` 的 `port` 或使用不同端口启动。
- CORS 错误：优先使用开发代理（`server.proxy`），或在后端开启 CORS。
- 构建资源路径错误：检查 Vite 的 `base` 配置，部署到子路径时需要调整。

开发提示
- 若需要调试模型返回的对话内容，可先通过 Postman 或 curl 调用后端 `/api/chat`，确认后端返回格式后再调整前端解析逻辑。
- 图谱可视化建议分离数据与渲染层：后端返回标准化节点/边结构（id、label、属性），前端仅负责渲染与交互。

参考
- 项目根目录 `README.md`：包含环境准备、数据准备、构建与后端启动等整体说明。
- `proj-docs/`：更详细的架构与数据处理文档（如有）。

如果你希望我把本 README 转为英文版、加入 `package.json` 示例脚本、或生成一个 `vite.config.js` 的完整样例文件，我可以继续补充。
