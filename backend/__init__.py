# Setup dynamic aliasing first for compatibility with this environment
import sys
try:
    import langchain_classic.retrievers
    import langchain_classic.retrievers.document_compressors
    sys.modules['langchain.retrievers'] = langchain_classic.retrievers
    sys.modules['langchain.retrievers.document_compressors'] = langchain_classic.retrievers.document_compressors
except Exception:
    pass
