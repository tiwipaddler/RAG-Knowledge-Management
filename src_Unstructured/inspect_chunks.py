import pickle

with open("cached_documents.pkl", "rb") as f:
    documents = pickle.load(f)

print(f"Total chunks: {len(documents)}\n")

# Count what categories of content exist
categories = {}
tables_with_html = 0
chunks_with_images = 0

for doc in documents:
    cat = doc.metadata.get("category", "unknown")
    categories[cat] = categories.get(cat, 0) + 1
    
    if "text_as_html" in doc.metadata:
        tables_with_html += 1
    
    if "image_base64" in doc.metadata or "image_mime_type" in doc.metadata:
        chunks_with_images += 1

print("Categories found:")
for cat, count in categories.items():
    print(f"  {cat}: {count}")

print(f"\nTables with HTML structure: {tables_with_html}")
print(f"Chunks with image data: {chunks_with_images}")

# Show all metadata keys present (helps see what's actually available)
print("\nAll metadata keys seen across chunks:")
all_keys = set()
for doc in documents:
    all_keys.update(doc.metadata.keys())
print(f"  {sorted(all_keys)}")