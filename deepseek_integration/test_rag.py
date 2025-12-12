from rag_system import RAGSystem

def test_rag():
    print("Initializing RAG System...")
    rag = RAGSystem()
    
    # Force ingest for the test to ensure data is fresh
    print("Ingesting document...")
    rag.ingest_document("公司制度.txt")
    
    test_queries = [
        "迟到怎么扣钱？",
        "婚假有多少天？",
        "试用期多久？",
        "如何在公司内部发文？"
    ]
    
    for q in test_queries:
        print(f"\nTesting Query: {q}")
        answer, sources = rag.query(q)
        print(f"Answer: {answer}")
        print("Sources:", [m['section'] for m in sources])

if __name__ == "__main__":
    test_rag()
