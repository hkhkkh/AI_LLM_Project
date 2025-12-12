import os
import re
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from api_client import DeepSeekClient
import sys

# Add parent directory to path to import global config if needed, 
# but we will try to be self-contained or use absolute paths.
# The embedding model path from the root config.py
LOCAL_EMBEDDING_PATH = "D:/AI_LLM_Project/models/modelscope/BAAI/bge-small-zh-v1___5"
ONLINE_EMBEDDING_MODEL = "BAAI/bge-small-zh-v1.5"

class DocumentProcessor:
    def __init__(self, file_path):
        self.file_path = file_path

    def parse_markdown(self):
        """
        Parses the markdown file and splits it into chunks based on headers.
        Returns a list of dictionaries with 'content' and 'metadata'.
        """
        with open(self.file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        chunks = []
        current_headers = {} # Level -> Header Text
        current_content = []
        
        # Regex to match headers like #, ##, ###
        header_pattern = re.compile(r'^(#+)\s+(.*)')

        def flush_chunk():
            if current_content:
                text = "".join(current_content).strip()
                if text:
                    # Construct context path from headers
                    # Sort headers by level to maintain hierarchy
                    sorted_levels = sorted(current_headers.keys())
                    path = " > ".join([current_headers[l] for l in sorted_levels])
                    
                    full_content = f"【{path}】\n{text}"
                    
                    chunks.append({
                        "content": full_content,
                        "metadata": {
                            "source": os.path.basename(self.file_path),
                            "section": path
                        }
                    })
            current_content.clear()

        for line in lines:
            match = header_pattern.match(line)
            if match:
                flush_chunk()
                level = len(match.group(1))
                title = match.group(2).strip()
                
                # Update current headers: clear any deeper levels
                current_headers = {k: v for k, v in current_headers.items() if k < level}
                current_headers[level] = title
            else:
                current_content.append(line)
        
        flush_chunk() # Flush the last chunk
        return chunks

class RAGSystem:
    def __init__(self, db_path="./chroma_db", collection_name="company_policy"):
        self.client = DeepSeekClient()
        
        # Initialize Embedding Model
        print("Loading embedding model...")
        try:
            if os.path.exists(LOCAL_EMBEDDING_PATH):
                print(f"Using local embedding model: {LOCAL_EMBEDDING_PATH}")
                self.embedding_model = SentenceTransformer(LOCAL_EMBEDDING_PATH)
            else:
                print(f"Local model not found. Downloading/Using: {ONLINE_EMBEDDING_MODEL}")
                self.embedding_model = SentenceTransformer(ONLINE_EMBEDDING_MODEL)
        except Exception as e:
            print(f"Error loading embedding model: {e}")
            print("Falling back to default sentence-transformers model...")
            self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

        # Initialize ChromaDB
        self.chroma_client = chromadb.PersistentClient(path=db_path)
        self.collection = self.chroma_client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )

    def ingest_document(self, file_path):
        """Process and index the document."""
        processor = DocumentProcessor(file_path)
        chunks = processor.parse_markdown()
        
        print(f"Found {len(chunks)} chunks. Indexing...")
        
        ids = []
        documents = []
        metadatas = []
        embeddings = []

        # Batch processing for embeddings to be efficient
        batch_size = 32
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i+batch_size]
            batch_texts = [c['content'] for c in batch]
            
            # Generate embeddings
            batch_embeddings = self.embedding_model.encode(batch_texts).tolist()
            
            for j, chunk in enumerate(batch):
                ids.append(f"doc_{i+j}")
                documents.append(chunk['content'])
                metadatas.append(chunk['metadata'])
                embeddings.append(batch_embeddings[j])

        # Add to ChromaDB
        if ids:
            self.collection.upsert(
                ids=ids,
                documents=documents,
                embeddings=embeddings,
                metadatas=metadatas
            )
        print("Indexing complete.")

    def query(self, user_query, n_results=3):
        """Retrieve context and generate answer."""
        # 1. Embed query
        query_embedding = self.embedding_model.encode([user_query]).tolist()
        
        # 2. Retrieve
        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=n_results
        )
        
        retrieved_docs = results['documents'][0]
        retrieved_metas = results['metadatas'][0]
        
        # 3. Construct Prompt
        context_str = "\n\n".join(retrieved_docs)
        
        system_prompt = """你是一个专业的公司制度咨询助手。
你的职责是根据提供的【公司制度上下文】回答员工关于公司制度的问题。

请遵循以下规则：
1. 如果用户询问你的身份（如“你是谁”），请回答：“我是您的公司制度咨询助手，可以为您解答关于考勤、休假、福利等公司制度的问题。”
2. 对于制度类问题，请严格根据【公司制度上下文】回答。
3. 如果上下文中没有相关信息，请直接回答“当前公司制度中未提及相关内容”，不要编造。
4. 回答要条理清晰，语气专业。引用相关制度时，请说明依据。"""
        
        user_prompt = f"""用户问题：{user_query}

【公司制度上下文】：
{context_str}

请根据以上上下文回答用户问题。"""

        # 4. Call LLM
        print(f"\nThinking... (Retrieving {n_results} chunks)")
        response = self.client.simple_chat(user_prompt, system_prompt=system_prompt)
        
        return response, retrieved_metas

if __name__ == "__main__":
    # Initialize System
    rag = RAGSystem()
    
    # Check if we need to ingest (simple check: if collection is empty)
    if rag.collection.count() == 0:
        print("Database empty. Ingesting document...")
        rag.ingest_document("公司制度.txt")
    else:
        print(f"Database loaded with {rag.collection.count()} documents.")
        # Optional: Force re-ingest if needed
        # rag.ingest_document("公司制度.txt")

    # Interactive Loop
    print("\n" + "="*50)
    print("公司制度 AI 助手 (输入 'exit' 退出)")
    print("="*50)
    
    while True:
        query = input("\n请输入您的问题: ")
        if query.lower() in ['exit', 'quit', 'q']:
            break
        
        if not query.strip():
            continue
            
        answer, sources = rag.query(query)
        print("\n" + "-"*30)
        print("【AI 回答】:")
        print(answer)
        print("-"*30)
        print("【参考来源】:")
        for meta in sources:
            print(f"- {meta['section']}")
