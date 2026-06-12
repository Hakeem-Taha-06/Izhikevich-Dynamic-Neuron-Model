import os
import PyPDF2

def extract_pdf(file_path):
    with open(file_path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        text = ""
        for page in reader.pages:
            t = page.extract_text()
            if t:
                text += t + "\n"
        out_path = file_path + ".txt"
        with open(out_path, "w", encoding="utf-8") as out_f:
            out_f.write(f"--- {os.path.basename(file_path)} ---\n")
            out_f.write(text)

extract_pdf("Paper/project-statement.pdf")
extract_pdf("Paper/Literature_review.pdf")
extract_pdf("Paper/mathematical_modeling.pdf")
