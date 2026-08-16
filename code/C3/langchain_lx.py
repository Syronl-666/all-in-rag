from llama_index.core import load_index_from_storage, Settings, StorageContext
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
#from llama_index.vector_stores.faiss import FaissVectorStore

# 配置嵌入模型
Settings.embed_model=HuggingFaceEmbedding("BAAI/bge-small-zh-v1.5")

# 加载本地向量库
persist_path = "./llamaindex_index_store"
storage = StorageContext.from_defaults(persist_dir = persist_path)
index = load_index_from_storage(storage_context = storage)

# 相似查询
retriever = index.as_retriever(similarity_top_k=1)
results = retriever.retrieve("llamaindex")

for result in results:
    print(f'最相似:\n{result.node.text}\n{result.score}')

'''
LlamaIndex是一个用于构建和查询私有或领域特定数据的框架。
0.5826540646935399
'''
