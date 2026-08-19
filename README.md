# SemiRAG

面向半导体与芯片领域知识问答的 RAG（检索增强生成）示例项目。项目使用 LangChain、LangGraph 和 Milvus，将 Markdown 知识库转为稠密向量与 BM25 稀疏向量，并提供两种命令行问答工作流。

## 功能

- 解析 Markdown 文档，并按语义切分后写入 Milvus。
- 结合 BGE 中文稠密向量、Milvus 内置 BM25 和 RRF 进行混合检索。
- `graph`：带检索工具调用、文档相关性判断和问题改写的 Agent 工作流。
- `graph2`：按问题路由到本地知识库或联网搜索，并对检索结果、答案相关性与事实一致性进行评估。
- 支持 Tavily 作为知识库外问题的联网搜索兜底。

## 项目结构

```text
.
├── RAG_PROJECT/
│   ├── agent/              # 具备检索工具的 Agent
│   ├── graph/              # 基础 Agentic RAG 工作流
│   ├── graph2/             # 自适应 RAG / 联网搜索工作流
│   ├── documents/          # Markdown 解析与 Milvus 写入
│   ├── llm_models/         # 聊天模型与嵌入模型配置
│   ├── tools/              # Milvus retriever 工具
│   ├── utils/              # 环境变量与日志工具
│   └── datas/md/           # 示例知识库文档
├── md/                     # 半导体技术 Markdown 语料
└── requirements.txt        # Python 依赖锁定列表
```

`RAG_PROJECT/datas/output/` 为 PDF 解析中间结果，不是运行时必需文件，已由 `.gitignore` 忽略。

## 环境要求

- Python 3.11（项目 IDE 配置使用该版本）。
- 可访问的 Milvus 2.5+ 服务，需支持内置 BM25 Function。
- 可用的 OpenAI 兼容聊天模型与嵌入模型 API。
- 使用联网搜索时，还需要 Tavily API Key。

## 安装

```bash
git clone git@github.com:YYFPS/SemiRAG.git
cd SemiRAG

python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Windows PowerShell 激活虚拟环境：

```powershell
.venv\Scripts\Activate.ps1
```

## 配置

在 `RAG_PROJECT/` 下创建 `.env`。该文件已经被 Git 忽略，切勿将真实密钥提交到仓库。

```dotenv
OPENAI_API_KEY=your_api_key
DEEPSEEK_API_KEY=                         # 仅在切换到 DeepSeek 配置时需要
TAVILY_API_KEY=your_tavily_api_key        # 使用联网搜索时需要
```

当前聊天模型和 OpenAI 嵌入模型在 `RAG_PROJECT/llm_models/` 中使用 `https://xiaoai.plus/v1` 作为 OpenAI 兼容接口；如使用其他服务商，请相应修改 `all_llm.py` 与 `embeddings_model.py`。

Milvus 地址与集合名在 `RAG_PROJECT/utils/env_utils.py` 中配置：

```python
MILVUS_URI = "http://<your-milvus-host>:19530"
COLLECTION_NAME = "t_collection01"
```

请先替换为自己可访问的服务地址。运行导入脚本会重建同名集合，因此不要将生产集合名直接用于示例脚本。

## 导入知识库

导入逻辑位于 `RAG_PROJECT/documents/write_milvus.py`。先修改其中的 `md_dir`，使其指向待导入的 Markdown 文件夹，例如本仓库根目录的 `md`：

```python
md_dir = r"/absolute/path/to/SemiRAG/md"
```

然后从 `RAG_PROJECT` 目录运行：

```bash
cd RAG_PROJECT
python -m documents.write_milvus
```

脚本会创建集合、用 `UnstructuredMarkdownLoader` 解析一级目录下的 `.md` 文件、按语义切分，并通过多进程写入 Milvus。它在同名集合已存在时会删除并重建该集合。

## 运行

以下命令均应从 `RAG_PROJECT` 目录执行。

### 基础 Agentic RAG

```bash
python -m graph.graph1
```

Agent 会判断是否调用 Milvus 检索工具；检索结果不相关时，会改写问题后重试。输入 `q`、`quit` 或 `exit` 结束会话。

### 自适应 RAG 与联网搜索

```bash
python -m graph2.graph_2
```

此工作流会将半导体材料、芯片制造和光刻等问题路由到 Milvus；其他问题会调用 Tavily。检索结果会经过相关性评分，生成答案还会经过事实一致性与答题有效性检查。

## 工作流概览

```text
用户问题
  ├─ graph：Agent → Milvus 检索 → 相关性判断 → 生成 / 改写后重试
  └─ graph2：问题路由 ─┬→ Milvus 混合检索 → 文档评分 → 生成 → 答案评估
                         └→ Tavily 搜索 ───────────────────→ 生成 → 答案评估
```

## 常见问题

| 现象 | 处理方式 |
| --- | --- |
| `ModuleNotFoundError` | 进入 `RAG_PROJECT` 后，以 `python -m graph.graph1` 或 `python -m graph2.graph_2` 运行，不要直接运行内部脚本文件。 |
| 无法连接 Milvus | 检查 `MILVUS_URI`、端口、网络可达性与 Milvus 版本。 |
| 联网搜索报错 | 在 `.env` 中配置 `TAVILY_API_KEY`，或仅提问知识库覆盖的半导体问题。 |
| 模型调用认证失败 | 检查 `OPENAI_API_KEY` 与 OpenAI 兼容接口配置；切换服务商时同步修改模型配置文件。 |
| 导入脚本找不到数据 | 修改 `write_milvus.py` 中的 `md_dir`；当前代码保留了开发机上的 Windows 示例路径。 |

## 注意事项

- `RAG_PROJECT/main.py` 是 PyCharm 生成的示例文件，并非 RAG 应用入口。
- 提交前请确认不包含 `.env`、密钥文件、虚拟环境、日志或解析中间产物；这些模式已写入 `.gitignore`。
- 仓库目前未包含许可证文件。对外使用或分发前，请先明确许可证。
