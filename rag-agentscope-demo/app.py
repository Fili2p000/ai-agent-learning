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
from dashscope import Generation
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
# AgentScope 初始化（只执行一次）
# ──────────────────────────────────────────────
@st.cache_resource
def init_agentscope():
    dashscope.api_key = DASHSCOPE_API_KEY
    return "ready"  # 占位，保持调用接口不变


# ──────────────────────────────────────────────
# 核心问答函数：RAG 检索 + Agent 生成
# ──────────────────────────────────────────────
def ask_agent(agent, question: str) -> str:
    context = query_knowledge(question)
    augmented_content = f"""【参考上下文】
{context}

【用户问题】
{question}"""

    response = Generation.call(
        model="qwen-plus",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": augmented_content},
        ],
        result_format="message",
    )
    return response.output.choices[0].message.content

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
            agent = init_agentscope()
        except Exception as e:
            st.error(f"AgentScope 初始化失败: {e}")
            st.stop()

        # 调用 RAG + Agent 生成回答
        with st.chat_message("assistant"):
            with st.spinner("正在检索知识库并生成回答..."):
                try:
                    answer = ask_agent(agent, prompt)
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
