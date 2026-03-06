# rag_utils.py
# 高速公路知识库 RAG 工具模块
# 负责：文本切片 → 向量化 → 存入 ChromaDB → 检索

import os
import chromadb
from chromadb.utils import embedding_functions
import dashscope
from dashscope import TextEmbedding

# ──────────────────────────────────────────────
# 全局配置（API Key 在 app.py 中统一设置）
# ──────────────────────────────────────────────
CHROMA_PERSIST_DIR = "./chroma_db"
COLLECTION_NAME = "highway_knowledge"
CHUNK_SIZE = 500       # 每个文本块的字符数
CHUNK_OVERLAP = 50     # 相邻块之间的重叠字符数
TOP_K = 3              # 检索返回的最相似片段数量
EMBED_MODEL = "text-embedding-v2"  # DashScope 嵌入模型


# ──────────────────────────────────────────────
# 1. 文本切片函数
# ──────────────────────────────────────────────
def split_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """
    将长文本按字符数切片，相邻片段有 overlap 字符的重叠。
    返回：片段列表 (list of str)
    """
    chunks = []
    start = 0
    text_len = len(text)

    while start < text_len:
        end = min(start + chunk_size, text_len)
        chunk = text[start:end].strip()
        if chunk:  # 过滤空块
            chunks.append(chunk)
        if end == text_len:
            break
        start += chunk_size - overlap  # 下一个块从 (start + chunk_size - overlap) 开始

    return chunks


# ──────────────────────────────────────────────
# 2. DashScope 向量化函数
# ──────────────────────────────────────────────
def embed_texts(texts: list[str]) -> list[list[float]]:
    """
    调用 DashScope TextEmbedding API 批量向量化文本。
    DashScope 单次最多处理 25 条，超过需分批。
    返回：嵌入向量列表
    """
    all_embeddings = []
    batch_size = 25  # DashScope 单批上限

    for i in range(0, len(texts), batch_size):
        batch = texts[i: i + batch_size]
        response = TextEmbedding.call(
            model=EMBED_MODEL,
            input=batch
        )
        if response.status_code != 200:
            raise RuntimeError(
                f"DashScope Embedding 调用失败: {response.code} - {response.message}"
            )
        # 按 text_index 顺序提取向量
        embeddings = sorted(
            response.output["embeddings"],
            key=lambda x: x["text_index"]
        )
        all_embeddings.extend([e["embedding"] for e in embeddings])

    return all_embeddings


# ──────────────────────────────────────────────
# 3. ChromaDB 初始化
# ──────────────────────────────────────────────
def get_chroma_collection():
    """
    获取（或创建）ChromaDB 集合。
    使用本地持久化存储，向量维度由 DashScope text-embedding-v2 决定（1536维）。
    """
    client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
    # get_or_create_collection：集合不存在时自动创建
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}  # 使用余弦相似度
    )
    return collection


# ──────────────────────────────────────────────
# 4. 文件加载 & 建库（核心入口函数）
# ──────────────────────────────────────────────
def build_knowledge_base(file_path: str) -> tuple[bool, str]:
    """
    读取文本文件，切片、向量化，并存入 ChromaDB。
    
    参数:
        file_path: 文本文件路径（支持 .txt）
    返回:
        (success: bool, message: str)
    """
    try:
        # 读取文件
        with open(file_path, "r", encoding="utf-8") as f:
            raw_text = f.read()

        if not raw_text.strip():
            return False, "文件内容为空，请检查文件。"

        # 切片
        chunks = split_text(raw_text)
        if not chunks:
            return False, "文本切片失败，请检查文件内容。"

        # 向量化
        embeddings = embed_texts(chunks)

        # 存入 ChromaDB（先清空旧数据，保证幂等）
        collection = get_chroma_collection()

        # 删除该 collection 内的所有旧文档（防止重复加载）
        existing = collection.get()
        if existing["ids"]:
            collection.delete(ids=existing["ids"])

        # 批量插入
        ids = [f"chunk_{i}" for i in range(len(chunks))]
        collection.add(
            ids=ids,
            documents=chunks,
            embeddings=embeddings,
            metadatas=[{"source": os.path.basename(file_path), "chunk_index": i}
                       for i in range(len(chunks))]
        )

        return True, f"✅ 知识库构建成功！共处理 {len(chunks)} 个文本片段。"

    except FileNotFoundError:
        return False, f"文件未找到: {file_path}"
    except Exception as e:
        return False, f"构建知识库时出错: {str(e)}"


# ──────────────────────────────────────────────
# 5. 检索函数（供 app.py 调用）
# ──────────────────────────────────────────────
def query_knowledge(question: str, top_k: int = TOP_K) -> str:
    """
    将问题向量化，从 ChromaDB 检索 Top-K 相似片段，拼接成上下文字符串。
    
    参数:
        question: 用户问题
        top_k: 返回的相似片段数量
    返回:
        拼接后的上下文字符串（供注入 Prompt）
    """
    try:
        # 问题向量化
        q_embeddings = embed_texts([question])

        # ChromaDB 检索
        collection = get_chroma_collection()
        results = collection.query(
            query_embeddings=q_embeddings,
            n_results=min(top_k, collection.count()),  # 避免请求数量超过实际文档数
            include=["documents", "distances", "metadatas"]
        )

        docs = results["documents"][0]
        distances = results["distances"][0]

        if not docs:
            return "（知识库中未检索到相关内容）"

        # 拼接上下文，标注片段编号和相似度
        context_parts = []
        for i, (doc, dist) in enumerate(zip(docs, distances)):
            similarity = round(1 - dist, 3)  # 余弦距离 → 相似度
            context_parts.append(
                f"【片段 {i+1}】（相似度: {similarity}）\n{doc}"
            )

        return "\n\n".join(context_parts)

    except Exception as e:
        return f"（检索时出错: {str(e)}）"


# ──────────────────────────────────────────────
# 6. 知识库状态检查
# ──────────────────────────────────────────────
def get_kb_status() -> dict:
    """
    返回知识库当前状态信息。
    """
    try:
        collection = get_chroma_collection()
        count = collection.count()
        return {
            "ready": count > 0,
            "chunk_count": count,
            "message": f"知识库已就绪，共 {count} 个文本片段" if count > 0
                       else "知识库为空，请先上传文件"
        }
    except Exception as e:
        return {
            "ready": False,
            "chunk_count": 0,
            "message": f"检查知识库状态失败: {str(e)}"
        }
