# Load your regulation text
pdf_path = "/data/RCentre.pdf"  # Your PDF file

raw_text = extract_pdf_text(pdf_path)
cleaned_text = clean_text(raw_text)

chunks = clause_aware_chunking(cleaned_text)

index, embeddings = create_vector_index(chunks)