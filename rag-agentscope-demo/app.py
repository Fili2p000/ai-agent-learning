# app.py
# 高速公路行业知识库问答 Demo
# 技术栈: AgentScope + DashScope(Qwen-Plus) + ChromaDB + Streamlit
#
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 快速启动步骤：
#   1. pip install -r requirements.txt
#   2. 将下方 DASHSCOPE_API_KEY 替换为你的真实 Key
#      （或设置环境变量: export DASHSCOPE_API_KEY="sk-xxx"）
#   3. streamlit run app.py
#   4. 在侧边栏上传 highway_rules.txt，点击「构建知识库」
#   5. 在聊天框输入问题即可！
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

import os
import tempfile
import streamlit as st
import dashscope
import asyncio
from dashscope import Generation
from agentscope.message import Msg
from agentscope.agent import AgentBase
from agentscope.model import DashScopeChatModel
from agentscope.message import Msg
# ── 内部模块 ──────────────────────────────────
from rag_utils import build_knowledge_base, query_knowledge, get_kb_status

# ══════════════════════════════════════════════
# ★ 在此填写你的 DashScope API Key ★
# ══════════════════════════════════════════════
DASHSCOPE_API_KEY = os.environ.get("DASHSCOPE_API_KEY", "sk-a01192ddcc234db9957126cb38e0895f")
#
# AgentScope 模型配置 ID（内部引用名，可自定义）
MODEL_CONFIG_NAME = "qwen_plus_chat"

# ──────────────────────────────────────────────
# 系统 Prompt：高速公路专家 Agent
# ──────────────────────────────────────────────
SYSTEM_PROMPT = """你是一个专业的高速公路行业专家助手。

【重要规则】
1. 请仅根据下方提供的【参考上下文】来回答用户问题。
2. 如果上下文中没有找到相关信息，请直接说"知识库中未找到相关信息"，不要编造任何内容。
3. 回答时请注明你所依据的是哪个片段（如"根据片段1..."）。
4. 使用清晰、专业的中文回答。
5. 如果用户的问题超出高速公路领域，礼貌地说明你只负责高速公路相关问答。"""




# ──────────────────────────────────────────────
# 定义Agent
# ──────────────────────────────────────────────

class AnswerAgent(AgentBase):
    def __init__(self, api_key: str):
        super().__init__()
        self.model = DashScopeChatModel(
            model_name="qwen-plus",
            api_key=api_key,
            stream=False,
        )

    async def reply(self, msg: Msg) -> Msg:
        response = await self.model(
            messages=[
                {"role": "system", "content": """你是一个专业的高速公路行业专家助手。

【重要规则】
1. 请根据提供的【知识库内容】和【网络搜索内容】回答问题。
2. 回答时必须注明每条信息的来源：
   - 知识库内容标注：📚 来自知识库
   - 网络搜索内容标注：🌐 来自网络搜索
3. 如果两个来源都没有相关信息，说明"未找到相关信息"，不要编造。
4. 使用清晰、专业的中文回答。"""},
                {"role": "user", "content": msg.content},
            ]
        )
        return Msg(name="高速公路专家", content=response['content'][0]['text'], role="assistant")

class JudgeAgent(AgentBase):
    def __init__(self, api_key: str):
        super().__init__()
        self.model = DashScopeChatModel(
            model_name="qwen-plus",
            api_key=api_key,
            stream=False,
        )

    async def reply(self, msg: Msg) -> Msg:
        response = await self.model(
            messages=[
                {"role": "system", "content": """你是一个信息充分性判断专家。
你会收到一个用户问题和从知识库检索到的内容。
请判断知识库内容是否足够回答用户问题。

请只返回以下 JSON 格式，不要有任何其他内容：
{"sufficient": true} 
或
{"sufficient": false, "search_query": "建议的搜索关键词"}"""},
                {"role": "user", "content": msg.content},
            ]
        )
        return Msg(name="判断专家", content=response['content'][0]['text'], role="assistant")



# ──────────────────────────────────────────────
# AgentScope 初始化（只执行一次）
# ──────────────────────────────────────────────
@st.cache_resource
def init_agentscope():
    return {
        "judge": JudgeAgent(api_key=DASHSCOPE_API_KEY),
        "answer": AnswerAgent(api_key=DASHSCOPE_API_KEY),
    }


# ──────────────────────────────────────────────
# 核心问答函数：RAG 检索 + Agent 生成
# ──────────────────────────────────────────────
def ask_agent(agents: dict, question: str) -> str:
    import json
    from ddgs import DDGS

    # Step 1: RAG 检索
    rag_context = query_knowledge(question)

    # Step 2: 判断 Agent 决定是否需要联网
    judge_msg = Msg(name="用户", role="user", content=f"""
用户问题：{question}

知识库内容：
{rag_context}
""")
    judge_response = asyncio.run(agents["judge"].reply(judge_msg))

    # Step 3: 解析判断结果
    search_context = ""
    try:
        judge_result = json.loads(judge_response.content)
        if not judge_result.get("sufficient", True):
            search_query = judge_result.get("search_query", question)
            results = DDGS().text(search_query, max_results=3)
            if results:
                search_parts = [f"- {r['title']}: {r['body'][:200]}" for r in results]
                search_context = "\n".join(search_parts)
    except json.JSONDecodeError:
        pass  # 解析失败就跳过联网搜索

    # Step 4: 整合回答 Agent
    answer_msg = Msg(name="用户", role="user", content=f"""
用户问题：{question}

📚 知识库内容：
{rag_context}

🌐 网络搜索内容：
{search_context if search_context else "（本次未进行联网搜索）"}
""")
    answer_response = asyncio.run(agents["answer"].reply(answer_msg))
    return answer_response.content

# ══════════════════════════════════════════════
# Streamlit 页面
# ══════════════════════════════════════════════
def main():
    st.set_page_config(
        page_title="高速公路知识库问答",
        page_icon="🛣️",
        layout="wide"
    )

    st.title("🛣️ 高速公路行业知识库问答 Demo")
    st.caption("基于 AgentScope + DashScope(Qwen-Plus) + ChromaDB 的 RAG 问答系统")

    # ── 侧边栏：知识库管理 ──────────────────────
    with st.sidebar:
        st.header("📂 知识库管理")

        # 知识库状态显示
        kb_status = get_kb_status()
        if kb_status["ready"]:
            st.success(kb_status["message"])
        else:
            st.warning(kb_status["message"])

        st.divider()

        # 文件上传区域
        st.subheader("上传知识文件")
        uploaded_file = st.file_uploader(
            "选择 .txt 文件",
            type=["txt"],
            help="支持 UTF-8 编码的文本文件"
        )

        if uploaded_file is not None:
            st.info(f"已选择: {uploaded_file.name}（{uploaded_file.size / 1024:.1f} KB）")

            if st.button("🔨 构建知识库", type="primary", use_container_width=True):
                # 将上传文件保存到临时路径
                with tempfile.NamedTemporaryFile(
                    mode="wb", suffix=".txt", delete=False
                ) as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_path = tmp_file.name

                # 调用 RAG 工具构建知识库
                with st.spinner("正在处理文件，请稍候..."):
                    success, message = build_knowledge_base(tmp_path)

                # 清理临时文件
                os.unlink(tmp_path)

                if success:
                    st.success(message)
                    st.rerun()  # 刷新页面更新知识库状态
                else:
                    st.error(message)

        st.divider()

        # 使用说明
        with st.expander("📖 使用说明"):
            st.markdown("""
**操作步骤：**
1. 上传 `highway_rules.txt` 等知识文件
2. 点击「构建知识库」等待处理完成
3. 在右侧聊天框中输入问题
4. 系统自动检索相关内容并回答

**示例问题：**
- 高速公路最高限速是多少？
- 哪些车辆禁止上高速？
- ETC 车道行驶速度限制是多少？
- 隧道内发生火灾如何逃生？
- 超限运输对路面有什么影响？
            """)

        # API Key 状态提示
        st.divider()
        if DASHSCOPE_API_KEY == "sk-your-dashscope-api-key-here":
            st.error("⚠️ 请在 app.py 中填写真实的 DASHSCOPE_API_KEY")
        else:
            st.success("✅ API Key 已配置")

    # ── 主区域：聊天界面 ────────────────────────
    # 初始化对话历史（存储在 session_state）
    if "messages" not in st.session_state:
        st.session_state.messages = []
        # 添加欢迎消息
        st.session_state.messages.append({
            "role": "assistant",
            "content": "您好！我是高速公路行业专家助手 🛣️\n\n"
                       "请先在左侧侧边栏上传知识文件并构建知识库，然后就可以向我提问了。\n\n"
                       "**示例问题：**\n"
                       "- 高速公路最高限速是多少？\n"
                       "- 哪些车辆禁止上高速？\n"
                       "- 如何处理高速公路上的车辆故障？"
        })

    # 显示历史消息
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # 聊天输入框
    if prompt := st.chat_input("请输入您的问题..."):

        # 检查前置条件
        if DASHSCOPE_API_KEY == "sk-your-dashscope-api-key-here":
            st.error("请先在 app.py 中配置有效的 DASHSCOPE_API_KEY！")
            st.stop()

        if not get_kb_status()["ready"]:
            st.warning("知识库为空！请先在侧边栏上传文件并点击「构建知识库」。")
            st.stop()

        # 显示用户消息
        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        # 初始化 Agent（缓存，只初始化一次）
        try:
            agents = init_agentscope()
        except Exception as e:
            st.error(f"AgentScope 初始化失败: {e}")
            st.stop()

        # 调用 RAG + Agent 生成回答
        with st.chat_message("assistant"):
            with st.spinner("正在检索知识库并生成回答..."):
                try:
                    answer = ask_agent(agents, prompt)
                    st.markdown(answer)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer
                    })
                except Exception as e:
                    error_msg = f"⚠️ 回答生成失败: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": error_msg
                    })

    # ── 底部工具栏 ──────────────────────────────
    col1, col2, col3 = st.columns([1, 1, 4])
    with col1:
        if st.button("🗑️ 清空对话"):
            st.session_state.messages = []
            st.rerun()
    with col2:
        kb = get_kb_status()
        st.caption(f"知识库: {'✅' if kb['ready'] else '❌'} {kb['chunk_count']} 片段")


if __name__ == "__main__":
    main()
