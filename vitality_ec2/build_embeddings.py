"""
build_embeddings.py  -  Vitality Leisure Park  -  True RAG Document Embedding
===============================================================================
Run ONCE locally before deploying:
    pip install cohere pypdf
    python3 build_embeddings.py

Requires both PDFs in the same folder:
    VST_KochWerk_Speisekarte_save.pdf
    FitnessClub_Kursplan_2026_01_v01d.pdf

What this does (true RAG - reads directly from PDF files):
    1. Reads both PDFs using pypdf - no hardcoding of content
    2. Splits pages into chunks
    3. Embeds each chunk with Cohere embed-english-v3.0 (1024-dim vectors)
    4. Saves chunks + vectors to embeddings.json
"""

import json, os, re
import cohere
import numpy as np
from pypdf import PdfReader

COHERE_API_KEY = os.environ.get("COHERE_API_KEY", "paste-your-key-here")
co = cohere.ClientV2(COHERE_API_KEY)

# ── Step 1: Extract text from PDFs ────────────────────────────────────────────
def extract_pages(path):
    reader = PdfReader(path)
    pages  = []
    for page in reader.pages:
        text = page.extract_text()
        if text and text.strip():
            text = re.sub(r'\n{3,}', '\n\n', text.strip())
            text = re.sub(r' {2,}', ' ', text)
            pages.append(text)
    return pages

print("Reading PDFs...")
menu_pages    = extract_pages("Menu_RAG.pdf")
fitness_pages = extract_pages("Fitness_RAG.pdf")
print(f"  Menu: {len(menu_pages)} pages")
print(f"  Fitness: {len(fitness_pages)} pages")

# ── Step 2: Chunk the pages ────────────────────────────────────────────────────
chunks = []

# Menu: one chunk per page (each page = one menu section)
MENU_LABELS = [
    "Welcome", "Soups and specials", "Snacks and Wraps",
    "Salads and Burgers", "Bowls and Main dishes", "Pizza",
    "Pasta and Desserts", "Hot drinks", "Cold drinks",
    "Beer", "Wine and Aperitifs", "Wine continued",
]
for i, text in enumerate(menu_pages):
    if i == 0 and "Herzlich willkommen" in text:
        continue  # skip pure intro page
    label = MENU_LABELS[i] if i < len(MENU_LABELS) else f"Menu section {i+1}"
    chunks.append({
        "id":      f"menu_page_{i+1}",
        "source":  "menu",
        "section": label,
        "text":    f"KochWerk Restaurant menu - {label}:\n{text}"
    })

# Fitness page 1: full timetable as one chunk
if fitness_pages:
    chunks.append({
        "id":      "fitness_timetable",
        "source":  "fitness",
        "section": "Weekly class timetable",
        "text":    f"FitnessClub weekly timetable with all classes and times:\n{fitness_pages[0]}"
    })

# Fitness page 2: class descriptions - split by class
if len(fitness_pages) > 1:
    desc_text = fitness_pages[1]
    # Split on lines that start a new class (capital word followed by | )
    parts = re.split(r'\n(?=[A-ZBEH][a-zA-ZäöüÄÖÜ\-]+(?:PUMP|BALANCE|board|Yoga|Work|Nordic|Aerobic|Intervall)?[®™]?\s*\|)', desc_text)
    for part in parts:
        part = part.strip()
        if len(part) > 40:
            name = re.match(r'^([^\|\n]+)', part)
            label = name.group(1).strip()[:40] if name else "Class"
            chunks.append({
                "id":      f"fitness_{re.sub(r'[^a-z0-9]', '_', label.lower()[:25])}",
                "source":  "fitness",
                "section": f"Class: {label}",
                "text":    f"FitnessClub class description - {part}"
            })

print(f"\nChunks created: {len(chunks)}")
for c in chunks:
    print(f"  [{c['source']:8s}] {c['id']}")

# ── Step 3: Embed with Cohere ──────────────────────────────────────────────────
print(f"\nEmbedding {len(chunks)} chunks...")
texts    = [c["text"] for c in chunks]
response = co.embed(
    texts=texts,
    model="embed-english-v3.0",
    input_type="search_document",
    embedding_types=["float"],
)
embeddings = response.embeddings.float_
print(f"Got {len(embeddings)} embeddings x {len(embeddings[0])} dims")

# Normalise for fast cosine similarity via dot product
emb_array = np.array(embeddings, dtype="float32")
norms     = np.linalg.norm(emb_array, axis=1, keepdims=True)
emb_norm  = (emb_array / norms).tolist()

# ── Step 4: Save ──────────────────────────────────────────────────────────────
with open("embeddings.json", "w", encoding="utf-8") as f:
    json.dump({
        "model":      "embed-english-v3.0",
        "n_chunks":   len(chunks),
        "chunks":     chunks,
        "embeddings": emb_norm,
    }, f, ensure_ascii=False)

print(f"\nSaved embeddings.json — {len(chunks)} chunks from 2 PDFs")
print("Now commit embeddings.json to GitHub.")
