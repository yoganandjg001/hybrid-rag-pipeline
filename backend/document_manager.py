import os
import re
import time
import hashlib
import json
from typing import List, Dict, Any, Tuple
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

MANIFEST_PATH = os.path.join("chroma_db", "manifest.json")

def get_file_hash(file_path: str) -> str:
    """Calculates MD5 checksum of a file for duplicate detection."""
    print(f"Calculating MD5 hash for {file_path}...")
    hasher = hashlib.md5()
    with open(file_path, 'rb') as f:
        buf = f.read(65536)
        while len(buf) > 0:
            hasher.update(buf)
            buf = f.read(65536)
    return hasher.hexdigest()

def load_manifest() -> Dict[str, Any]:
    """Loads metadata manifest mapping ingested filenames to details."""
    print(f"Loading manifest from {MANIFEST_PATH}...")
    if os.path.exists(MANIFEST_PATH):
        try:
            with open(MANIFEST_PATH, 'r') as f:
                return json.load(f)
        except Exception:
            return {}
    return {}

def save_manifest(manifest: Dict[str, Any]):
    """Saves metadata manifest to disk."""
    print(f"Saving manifest to {MANIFEST_PATH}...")
    os.makedirs(os.path.dirname(MANIFEST_PATH), exist_ok=True)
    with open(MANIFEST_PATH, 'w') as f:
        json.dump(manifest, f, indent=4)

def character_based_split(documents: List[Document], chunk_size: int = 500, chunk_overlap: int = 80) -> List[Document]:
    """Splits documents using RecursiveCharacterTextSplitter with configurable chunk_size and chunk_overlap."""
    print("Performing character-based splitting...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    split_docs = text_splitter.split_documents(documents)
    
    # Adjust section header and metadata in the character split chunks
    for idx, chunk in enumerate(split_docs):
        page_num = chunk.metadata.get("page", 0) + 1
        chunk.metadata["page"] = page_num
        chunk.metadata["section"] = f"Page {page_num} Chunks"
        chunk.metadata["source"] = os.path.basename(chunk.metadata.get("source", "document.pdf"))
        
    return split_docs

def structure_based_split(documents: List[Document], min_chunk_chars=500, max_chunk_chars=500) -> List[Document]:
    print("Performing structure-based splitting...")
    full_text = ""
    page_offsets = []
    
    for doc in documents:
        start = len(full_text)
        full_text += doc.page_content + "\n"
        end = len(full_text)
        page_offsets.append((doc.metadata.get("page", 0) + 1, start, end))
        
    header_pattern = re.compile(r'(?m)^\s*(\d+(\.\d+)*\s+[A-Z][^\n]{2,80})$')
    matches = list(header_pattern.finditer(full_text))
    
    if not matches:
        header_pattern_fallback = re.compile(r'(?m)^\s*([A-Z][A-Z\s]{4,60})\s*$')
        matches = list(header_pattern_fallback.finditer(full_text))
        
    if not matches:
        chunks = []
        for doc in documents:
            page_num = doc.metadata.get("page", 0) + 1
            chunks.append(Document(
                page_content=doc.page_content.strip(),
                metadata={
                    "section": f"Page {page_num}", 
                    "page": page_num, 
                    "source": os.path.basename(doc.metadata.get("source", "document.pdf"))
                }
            ))
        return chunks
        
    chunks = []
    
    # Capture any preamble text before the first section match (e.g. cover page or letter)
    if matches and matches[0].start() > 0:
        preamble_text = full_text[0:matches[0].start()].strip()
        if len(preamble_text) >= min_chunk_chars:
            chunks.append(Document(
                page_content=preamble_text,
                metadata={
                    "section": "Preamble / Policy Information",
                    "page": 1,
                    "source": os.path.basename(documents[0].metadata.get("source", "document.pdf"))
                }
            ))

    for i, m in enumerate(matches):
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(full_text)
        section_title = m.group(1).strip()
        section_text = full_text[start:end].strip()
        
        start_page = 1
        for page_num, p_start, p_end in page_offsets:
            if p_start <= start < p_end:
                start_page = page_num
                break
                
        if chunks and len(section_text) < min_chunk_chars:
            chunks[-1].page_content += "\n\n" + section_text
        else:
            chunks.append(Document(
                page_content=section_text,
                metadata={
                    "section": section_title, 
                    "page": start_page, 
                    "source": os.path.basename(documents[0].metadata.get("source", "document.pdf"))
                }
            ))
            
    char_splitter = RecursiveCharacterTextSplitter(chunk_size=max_chunk_chars, chunk_overlap=80)
    print(f"Splitting sections into chunks with max {max_chunk_chars} characters...")
    final_chunks = []
    for chunk in chunks:
        if len(chunk.page_content) > max_chunk_chars:
            sub_texts = char_splitter.split_text(chunk.page_content)
            for j, sub_text in enumerate(sub_texts):
                final_chunks.append(Document(
                    page_content=sub_text,
                    metadata={
                        **chunk.metadata,
                        "section": f"{chunk.metadata['section']} (Part {j+1})"
                    }
                ))
        else:
            final_chunks.append(chunk)
            
    return final_chunks

def ingest_pdf(
    file_path: str, 
    chunking_strategy: str = "structure", 
    chunk_size: int = None, 
    chunk_overlap: int = None, 
    force: bool = False
) -> Tuple[str, str]:
    """
    Ingests a PDF file incrementally into the RAG pipeline.
    Performs duplicate content check (MD5) and filename collision checks.
    Supports 'structure' and 'character' chunking strategies with custom sizes.
    Returns (status, filename) where status can be 'success', 'duplicate', or 'collision'.
    """
    print(f"Ingesting PDF: {file_path} with strategy '{chunking_strategy}'...")
    import backend.state as state
    state.initialize_models()
    
    filename = os.path.basename(file_path)
    file_hash = get_file_hash(file_path)
    manifest = load_manifest()
    
    # Resolve fallback default values
    if chunk_size is None:
        chunk_size = 500 if chunking_strategy == "structure" else 500
    if chunk_overlap is None:
        chunk_overlap = 80
        
    # 1. Duplicate Content Check (hash check)
    print(f"1. Checking for duplicates in manifest...")
    for existing_name, info in manifest.items():
        if info.get("hash") == file_hash and not force:
            return "duplicate", existing_name
            
    # 2. Filename Collision Check
    print(f"2. Checking for filename collisions in manifest...")
    if filename in manifest and not force:
        return "collision", filename
        
    # Copy file to persistent doc directory
    doc_dir = "documents"
    os.makedirs(doc_dir, exist_ok=True)
    persistent_path = os.path.join(doc_dir, filename)
    
    if os.path.abspath(file_path) != os.path.abspath(persistent_path):
        import shutil
        shutil.copy(file_path, persistent_path)
        
    # 3. Parse and chunk
    print(f"3. Parsing and chunking PDF...")
    loader = PyPDFLoader(persistent_path)
    docs = loader.load()
    
    if chunking_strategy == "character":
        new_chunks = character_based_split(docs, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    else:
        new_chunks = structure_based_split(docs, max_chunk_chars=chunk_size)
    
    # 4. Incremental Indexing (Chroma)
    print(f"4. Indexing chunks into Chroma...")
    if state.vectordb is None:
        state.vectordb = Chroma.from_documents(
            documents=new_chunks,
            embedding=state.embeddings,
            persist_directory="chroma_db"
        )
    else:
        # If forcing overwrite on a collision, delete old chunks from Chroma first
        if filename in manifest:
            state.vectordb.delete(where={"source": filename})
            
        state.vectordb.add_documents(new_chunks)
        
    print(f"5. Saving/Updating manifest...")
    manifest[filename] = {
        "chunk_count": len(new_chunks),
        "hash": file_hash,
        "chunking_strategy": chunking_strategy,
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "added_at": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    save_manifest(manifest)
    
    # 6. Rebuild in-memory retrievers (BM25, Hybrids)
    print(f"6. Rebuilding in-memory retrievers...")
    state.rebuild_from_persistent_files()
    
    return "success", filename

def delete_document(filename: str) -> bool:
    """
    Deletes an ingested document from Chroma, removes its persistent file,
    and updates BM25 indexes and manifest metadata.
    """
    import backend.state as state
    state.initialize_models()
    
    manifest = load_manifest()
    if filename not in manifest:
        return False
        
    # 1. Delete from Chroma
    if state.vectordb is not None:
        state.vectordb.delete(where={"source": filename})
        
    # 2. Delete physical document
    doc_path = os.path.join("documents", filename)
    if os.path.exists(doc_path):
        try:
            os.remove(doc_path)
        except Exception as e:
            print(f"Error removing physical document {filename}: {e}")
            
    # 3. Remove from manifest
    del manifest[filename]
    save_manifest(manifest)
    
    # 4. Rebuild in-memory state
    state.rebuild_from_persistent_files()
    
    # If no files left, clear vectordb
    if not manifest:
        state.vectordb = None
        
    return True
