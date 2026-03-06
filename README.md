# AI Agent Learning

个人学习项目，探索基于 LangGraph 和 AgentScope 框架的 AI Agent 开发。

---

## 项目结构

```
ai-agent-learning/
├── langgraph-demo/        # 基于 LangGraph 的 Agent 实现
└── rag-agentscope-demo/   # 基于 AgentScope 的 RAG 实现
```

---

## Demo 介绍

### 1. LangGraph Demo

基于 [LangGraph](https://github.com/langchain-ai/langgraph) 框架，实现了两个 Agent：

**Calculator Agent** (`calculator.py`)
- 使用 Claude Sonnet 作为推理核心
- 通过 Tool Calling 实现加减乘除运算
- 演示了 LangGraph 的基础 ReAct 循环：LLM → Tool → LLM

**Email Agent** (`main.py`)
- 使用 Gemini 2.5 Flash 作为推理核心
- 自动对邮件进行意图分类（问题 / Bug / 账单 / 功能请求）和紧急程度判断
- 根据分类结果路由到不同处理节点（文档搜索 / Bug 追踪 / 人工审核）
- 使用 `interrupt()` 实现 Human-in-the-loop，高风险邮件暂停等待人工确认后再发送
- 演示了 LangGraph 的条件路由、状态管理和持久化 Checkpointing
- **注意：邮件收发和 Human Interrupt 均为模拟实现，未接入真实邮件接口**

**运行方式：**

在 `langgraph-demo/` 目录下创建 `.env` 文件，填入以下 API Key：
```
ANTHROPIC_API_KEY=your_anthropic_api_key
GOOGLE_API_KEY=your_google_api_key
```

然后执行：
```bash
uv sync
uv run calculator.py
uv run main.py
```

---

### 2. AgentScope RAG Demo

基于 [AgentScope](https://github.com/modelscope/agentscope) 框架，结合 DashScope（Qwen-Plus）和 ChromaDB，实现了一个支持自定义文档上传的单 Agent RAG 基础系统：

- 使用 DashScope Embedding 对文档进行向量化，存入 ChromaDB
- 用户提问后，检索 Top-3 相似文本片段注入 Prompt，由 Qwen-Plus 生成回答
- 支持通过 Streamlit 启动交互式 Web 界面，上传自定义知识库文件
- 演示了完整的 RAG 流程：文档切分 → 向量化 → 检索 → 增强生成

详细运行说明见 `rag-agentscope-demo/README.md`。

---

## 技术栈

| 类别 | 技术 |
|------|------|
| Agent 框架 | LangGraph, AgentScope |
| LLM | Claude Sonnet, Gemini 2.5 Flash, Qwen-Plus |
| Embedding | DashScope text-embedding-v2 |
| 向量数据库 | ChromaDB |
| Web UI | Streamlit |
| 包管理 | uv |
