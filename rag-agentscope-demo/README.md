# 🛣️ 高速公路行业知识库问答 Demo

基于 **AgentScope + DashScope (Qwen-Plus) + ChromaDB + Streamlit** 构建的本地 RAG 问答系统。

## 📁 文件结构

```
highway_rag_demo/
├── app.py              # 主程序（Streamlit 界面 + AgentScope Agent）
├── rag_utils.py        # RAG 工具模块（切片、向量化、ChromaDB）
├── requirements.txt    # 依赖包
├── highway_rules.txt   # 示例知识文件（可替换为你自己的）
└── README.md           # 本文件
```

## 🚀 快速启动（5 分钟内跑起来）

### 第 1 步：安装依赖
```bash
pip install -r requirements.txt
```

### 第 2 步：填写 API Key
打开 `app.py`，找到第 28 行，将 API Key 替换为你的真实 Key：
```python
DASHSCOPE_API_KEY = "sk-your-actual-key-here"
```

**或者**使用环境变量（更安全）：
```bash
export DASHSCOPE_API_KEY="sk-your-actual-key-here"
```

> 💡 DashScope API Key 申请地址：https://dashscope.console.aliyun.com/

### 第 3 步：启动应用
```bash
streamlit run app.py
```

浏览器会自动打开 http://localhost:8501

### 第 4 步：使用流程
1. 在**左侧侧边栏**点击「Browse files」上传 `highway_rules.txt`
2. 点击「🔨 构建知识库」按钮，等待处理完成（约 10-30 秒）
3. 在**右侧聊天框**输入问题，例如：
   - 高速公路最高限速是多少？
   - ETC 车道行驶速度限制？
   - 隧道内发生火灾如何逃生？

## 🏗️ 架构说明

```
用户问题
   ↓
[DashScope Embedding] 问题向量化
   ↓
[ChromaDB] 余弦相似度检索 Top-3 片段
   ↓
[Prompt 注入] 上下文 + 问题 → 构造 Augmented Prompt
   ↓
[AgentScope DialogAgent] 调用 Qwen-Plus 生成回答
   ↓
[Streamlit] 显示回答
```

## ⚙️ 关键参数调整

在 `rag_utils.py` 顶部可调整：
| 参数 | 默认值 | 说明 |
|------|--------|------|
| `CHUNK_SIZE` | 500 | 每个文本块的字符数 |
| `CHUNK_OVERLAP` | 50 | 相邻块重叠字符数 |
| `TOP_K` | 3 | 检索返回的相似片段数 |
| `EMBED_MODEL` | text-embedding-v2 | DashScope 嵌入模型 |

## 🔧 常见问题

**Q: 出现 `chromadb` 相关错误？**
```bash
pip install chromadb --upgrade
```

**Q: DashScope 调用失败？**
- 检查 API Key 是否正确
- 确认账户余额充足
- 检查网络是否可访问 dashscope.aliyuncs.com

**Q: 想换成自己的知识文件？**
- 直接在侧边栏上传你的 .txt 文件即可
- 支持任意 UTF-8 编码的文本

**Q: ChromaDB 数据存在哪里？**
- 持久化在当前目录的 `./chroma_db/` 文件夹
- 删除该文件夹可清空知识库
