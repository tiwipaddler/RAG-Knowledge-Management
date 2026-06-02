import os
import glob
import pickle
from dotenv import load_dotenv
# Becasue of MacOS13 limitations
from langchain_unstructured import UnstructuredLoader

load_dotenv()


def load_documents_elements(docs_folder="docs", test_mode=True):
    """Load PDFs as raw elements (no chunking) to inspect what gets detected"""
    file_paths = glob.glob(os.path.join(docs_folder, "*.pdf"))

    if test_mode:
        file_paths = file_paths[:1]
        print(f"🧪 TEST MODE: processing 1 file only")

    print(f"Processing {len(file_paths)} file(s): {file_paths}")

    loader = UnstructuredLoader(
        file_path=file_paths,
        api_key=os.getenv("UNSTRUCTURED_API_KEY"),
        partition_via_api=True,
        strategy="hi_res",                                        # required for tables/images/formulas
        infer_table_structure=True,                               # tables as HTML
        extract_image_block_types=["Image", "Table", "Formula"],  # detect these as image regions
        extract_image_block_to_payload=True,                      # capture the actual image data
    )
    documents = loader.load()
    print(f"✅ Loaded {len(documents)} raw elements")
    return documents


def main():
    documents = load_documents_elements("docs", test_mode=True)

    # Cache so we don't re-call the API while inspecting
    with open("cached_elements.pkl", "wb") as f:
        pickle.dump(documents, f)
    print("✅ Cached raw elements to cached_elements.pkl")

    # Count the element types that came back
    types = {}
    for doc in documents:
        t = doc.metadata.get("category", "unknown")
        types[t] = types.get(t, 0) + 1

    print("\nElement types detected:")
    for t, count in sorted(types.items()):
        print(f"  {t}: {count}")

    # Write a readable txt inspection file
    with open("elements_inspection.txt", "w") as f:
        f.write(f"Total elements: {len(documents)}\n")
        f.write("="*60 + "\n\n")
        f.write("ELEMENT TYPE COUNTS:\n")
        for t, count in sorted(types.items()):
            f.write(f"  {t}: {count}\n")
        f.write("\n" + "="*60 + "\n\n")

        for i, doc in enumerate(documents):
            category = doc.metadata.get("category", "unknown")
            has_html = "text_as_html" in doc.metadata
            has_image = "image_base64" in doc.metadata
            page = doc.metadata.get("page_number", "?")

            f.write(f"--- ELEMENT {i+1} | type={category} | page={page} ---\n")
            f.write(f"  has_table_html: {has_html}\n")
            f.write(f"  has_image_data: {has_image}\n")
            f.write(f"  text ({len(doc.page_content)} chars): {doc.page_content[:500]}\n")
            if has_html:
                f.write(f"  TABLE HTML: {doc.metadata['text_as_html'][:500]}\n")
            f.write("\n")

    print("✅ Wrote readable inspection to elements_inspection.txt")


if __name__ == "__main__":
    main()

