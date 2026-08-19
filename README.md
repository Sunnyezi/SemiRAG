# SemiRAG

面向半导体与芯片领域知识问答的 Agentic RAG 示例。项目使用 LangChain、LangGraph 和 Milvus，将 Markdown 语料导入混合检索知识库，并提供基础 Agentic RAG 与自适应 RAG 两种交互式工作流。

## 目录结构

项目借鉴 RAGFlow 按职责分层的思路，但保持适合单体 Python 项目的轻量结构：

```text
SemiRAG/
├── src/semirag/                 # 唯一应用包
│   ├── agents/                  # 工具调用 Agent
│   ├── ingestion/               # Markdown 解析与 Milvus 导入
│   ├── models/                  # 聊天模型与嵌入模型
│   ├── retrieval/               # 混合检索工具
│   ├── utils/                   # 日志与图可视化辅助
│   └── workflows/               # LangGraph 工作流
│       ├── agentic/             # 基础工具调用 RAG
│       └── adaptive/            # 路由、改写、评分与联网兜底 RAG
├── data/
│   ├── knowledge_base/          # 可版本控制的 Markdown 语料
│   │   ├── semiconductor/       # 半导体技术语料
│   │   └── milvus_reference/    # Milvus 参考语料
│   ├── samples/                 # 小型输入样例
│   └── processed/               # 可再生解析产物（Git 忽略）
├── docs/
│   ├── assets/                  # 工作流图
│   └── references/              # 项目参考资料
├── .env.example                 # 可提交的环境变量模板
├── pyproject.toml               # 包与命令行入口定义
└── requirements.txt             # 运行依赖
```

## 功能

- 用 `UnstructuredMarkdownLoader` 解析 Markdown，并通过语义切分写入 Milvus。
- 使用 BGE 中文稠密向量、Milvus 内置 BM25 和 RRF 进行混合检索。
- `agentic` 工作流：模型调用检索工具，检索结果不相关时改写问题后重试。
- `adaptive` 工作流：将问题路由到本地知识库或 Tavily 搜索，并进行文档相关性、事实一致性和答题有效性评估。

## 环境要求

- Python 3.11。依赖版本与原项目的 IDE 配置均基于该版本；不建议直接使用 3.12+。
- 可访问的 Milvus 2.5+ 服务，且支持内置 BM25 Function。
- OpenAI 兼容的聊天与嵌入模型接口。
- 使用自适应工作流的联网搜索时，需要 Tavily API Key。

## 安装

```bash
git clone git@github.com:YYFPS/SemiRAG.git
cd SemiRAG

python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
```

Windows PowerShell：

```powershell
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pip install -e .
```

## 配置

首次使用时，从模板创建本地配置：

```bash
cp .env.example .env
```

`.env` 已被 Git 忽略，填入本地密钥和连接配置即可：

```dotenv
OPENAI_API_KEY=your_openai_compatible_key
DEEPSEEK_API_KEY=your_deepseek_key            # 切换到 DeepSeek 后需要
TAVILY_API_KEY=your_tavily_key                # 仅联网搜索需要

MILVUS_URI=http://localhost:19530
COLLECTION_NAME=semirag_knowledge_base
KNOWLEDGE_BASE_DIR=data/knowledge_base/semiconductor
```

默认聊天模型和嵌入模型的 OpenAI 兼容接口配置在 `src/semirag/models/`。若改用 DeepSeek，请在 `all_llm.py` 中启用对应配置；嵌入模型仍需要一个兼容的嵌入接口。

## 导入知识库

默认导入目录由 `KNOWLEDGE_BASE_DIR` 指定。导入前请确认 `MILVUS_URI`、集合名和语料路径正确：

```bash
semirag-ingest
```

导入命令会删除并重建同名 Milvus 集合，然后以多进程方式导入该目录一级的 `.md` 文件。不要直接对生产集合运行此命令。

## 运行

```bash
# 基础 Agentic RAG
semirag agentic

# 自适应 RAG：本地检索、问题改写、质量评分与联网兜底
semirag adaptive
```

也可以直接使用独立命令：

```bash
semirag-agentic
semirag-adaptive
```

输入 `q`、`quit` 或 `exit` 结束交互会话。

## 工作流概览

```text
用户问题
  ├─ agentic：Agent → Milvus 检索 → 相关性判断 → 生成 / 改写后重试
  └─ adaptive：路由 ─┬→ Milvus 混合检索 → 文档评分 → 生成 → 答案评估
                       └→ Tavily 搜索 ───────────────────→ 生成 → 答案评估
```

## 常见问题

| 现象 | 处理方式 |
| --- | --- |
| 找不到 `semirag` 命令 | 先激活虚拟环境并执行 `python -m pip install -e .`。 |
| 无法连接 Milvus | 检查 `.env` 的 `MILVUS_URI`、端口和 Milvus 版本。 |
| 联网搜索报错 | 在 `.env` 配置 `TAVILY_API_KEY`，或仅提问知识库覆盖的问题。 |
| 模型认证失败 | 检查 `OPENAI_API_KEY`；启用 DeepSeek 时还需修改 `src/semirag/models/all_llm.py`。 |
| 导入后没有文档 | 确认 `KNOWLEDGE_BASE_DIR` 指向含 Markdown 文件的目录；当前导入器不递归扫描子目录。 |

## 注意事项

- `data/processed/` 是可再生的解析中间结果，已被 `.gitignore` 忽略。
- `data/knowledge_base/milvus_reference/` 中原先引用但未随项目提供的图片，已替换为说明文字；不影响 Markdown 入库或问答能力。
- 不要提交 `.env`、私钥、虚拟环境、日志或本地 Milvus 数据。
- 本仓库尚未声明许可证；对外使用或分发前请先明确许可证。
