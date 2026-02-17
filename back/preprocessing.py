import re
import pdfplumber
from typing import List, Dict


# ==============================
# 1. PDF EXTRACTION + HEADER REMOVAL
# ==============================

def extract_pdf_text(pdf_path: str) -> str:
    """
    Extract text from PDF while removing repeated headers and page numbers.
    """

    full_text = ""

    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()

            if not text:
                continue

            lines = text.split("\n")
            cleaned_lines = []

            for line in lines:
                line = line.strip()

                # ---- REMOVE REPEATED HEADER (CUSTOMISE IF NEEDED) ----
                if "Regulations on the Use of the UTM Resource Centre" in line:
                    continue

                if "July 2011" in line or "V 1.3" in line:
                    continue

                # ---- REMOVE STANDALONE PAGE NUMBERS ----
                if re.fullmatch(r"\d+", line):
                    continue

                cleaned_lines.append(line)

            page_text = "\n".join(cleaned_lines)
            full_text += page_text + "\n"

    return full_text


# ==============================
# 2. TEXT CLEANING (YOUR ORIGINAL FUNCTION - IMPROVED)
# ==============================

def clean_text(text: str) -> str:
    """
    Remove formatting artefacts but preserve clause numbering.
    """

    # Remove excessive blank lines
    text = re.sub(r'\n+', '\n', text)

    # Remove excessive spaces (but keep single spaces)
    text = re.sub(r'[ \t]+', ' ', text)

    # Ensure regulation numbers stay on new line (e.g., "1.", "2.")
    text = re.sub(r'\s*(\d+\.)\s*', r'\n\1 ', text)

    return text.strip()


# ==============================
# 3. CLAUSE-AWARE CHUNKING (UNCHANGED LOGIC)
# ==============================

def clause_aware_chunking(text: str) -> List[Dict]:
    """
    Chunk by Regulation and Clause level only.
    Sub-clauses (a), (b), (c) remain inside their parent clause.
    """

    chunks = []

    # --- Split by Regulation ---
    regulation_pattern = r'(?=\n\s*\d+\.\s)'
    regulations = re.split(regulation_pattern, text)

    for reg in regulations:
        reg = reg.strip()
        if not reg:
            continue

        reg_match = re.match(r'(\d+)\.', reg)
        if not reg_match:
            continue

        reg_number = reg_match.group(1)

        # Remove regulation number from text body
        reg_body = re.sub(r'^\d+\.\s*', '', reg).strip()

        # --- Split by Clause Level (i), (ii), (iii) ---
        clause_pattern = r'(?=\(\s*[ivx]+\s*\))'
        clauses = re.split(clause_pattern, reg_body, flags=re.IGNORECASE)

        for clause in clauses:
            clause = clause.strip()
            if not clause:
                continue

            clause_match = re.match(r'\(\s*([ivx]+)\s*\)', clause, flags=re.IGNORECASE)
            clause_id = clause_match.group(1) if clause_match else "main"

            # Keep full clause including sub-clauses
            clause_text = clause.strip()

            if len(clause_text) > 50:
                chunks.append({
                    "regulation": reg_number,
                    "clause": clause_id,
                    "text": clause_text
                })

    return chunks