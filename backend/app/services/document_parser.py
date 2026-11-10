import re
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import fitz  # PyMuPDF
import docx

from backend.app.schemas.section import ResearchSectionSchema

CANONICAL_SECTIONS = [
    ("title", "Title", "A new perspective on research"),
    ("authors", "Authors", "Researcher Name, Co-author Name"),
    ("affiliations", "Affiliations", "Department of Research, University"),
    ("abstract", "Abstract", "A concise summary of the research question, approach and key findings."),
    ("introduction", "Introduction", "Introduce the topic and relevant background."),
    ("problem", "Problem statement", "Describe the challenge this work addresses."),
    ("objectives", "Objectives", "List the primary aims of this research."),
    ("methodology", "Methodology", "Explain the methods, materials and process."),
    ("results", "Results", "Present the key observations and data."),
    ("discussion", "Discussion", "Interpret the findings and their implications."),
    ("conclusion", "Conclusion", "Summarize the main takeaway."),
    ("references", "References", "Add sources cited in this poster."),
    ("keywords", "Keywords", "research, science, innovation"),
]

SECTION_PATTERNS = {
    "abstract": re.compile(r"^(abstract)\b", re.IGNORECASE),
    "introduction": re.compile(r"^([1I][\.\s]+)?(introduction)\b", re.IGNORECASE),
    "problem": re.compile(r"^(problem\s+statement|problem\s+formulation|problem\s+definition)\b", re.IGNORECASE),
    "objectives": re.compile(r"^(objectives|contributions|research\s+goals|goals)\b", re.IGNORECASE),
    "methodology": re.compile(r"^([2-4II-IV]+[\.\s]+)?(methodology|methods|proposed\s+(method|approach|system)|architecture|system\s+design)\b", re.IGNORECASE),
    "results": re.compile(r"^([3-5III-V]+[\.\s]+)?(results|experimental\s+results|experiments|evaluation|findings)\b", re.IGNORECASE),
    "discussion": re.compile(r"^([4-6IV-VI]+[\.\s]+)?(discussion|analysis)\b", re.IGNORECASE),
    "conclusion": re.compile(r"^([5-7V-VII]+[\.\s]+)?(conclusion|concluding\s+remarks|conclusions)\b", re.IGNORECASE),
    "references": re.compile(r"^(references|bibliography)\b", re.IGNORECASE),
    "keywords": re.compile(r"^(keywords|index\s+terms)\b", re.IGNORECASE),
}

class DocumentParserService:
    @classmethod
    def parse_pdf(cls, file_path: str) -> Tuple[Dict[str, Any], List[ResearchSectionSchema]]:
        doc = fitz.open(file_path)
        meta = doc.metadata or {}
        page_count = len(doc)
        
        extracted_sections: Dict[str, Dict[str, Any]] = {}
        for sec_id, default_title, default_content in CANONICAL_SECTIONS:
            extracted_sections[sec_id] = {
                "title": default_title,
                "content": "",
                "page": None
            }

        # Title heuristic: metadata or largest font on page 0
        doc_title = meta.get("title", "").strip()
        doc_author = meta.get("author", "").strip()
        
        current_section = None
        
        for page_idx in range(min(page_count, 15)):  # Process up to 15 pages
            page = doc[page_idx]
            page_num = page_idx + 1
            blocks = page.get_text("blocks")  # (x0, y0, x1, y1, text, block_no, block_type)
            
            for block in blocks:
                if block[6] != 0:  # Skip image blocks
                    continue
                raw_text = block[4].strip()
                if not raw_text:
                    continue

                lines = raw_text.split("\n")
                first_line = lines[0].strip()

                # Check if first page and title empty
                if page_idx == 0 and not doc_title and len(raw_text) < 200 and not SECTION_PATTERNS["abstract"].match(first_line):
                    doc_title = raw_text.replace("\n", " ")
                    extracted_sections["title"]["content"] = doc_title
                    extracted_sections["title"]["page"] = 1
                    continue

                # Match against section patterns
                matched_sec = None
                for sec_key, pattern in SECTION_PATTERNS.items():
                    if pattern.match(first_line):
                        matched_sec = sec_key
                        break
                
                if matched_sec:
                    current_section = matched_sec
                    remaining_text = "\n".join(lines[1:]).strip()
                    if extracted_sections[current_section]["page"] is None:
                        extracted_sections[current_section]["page"] = page_num
                    if remaining_text:
                        existing = extracted_sections[current_section]["content"]
                        extracted_sections[current_section]["content"] = f"{existing}\n{remaining_text}".strip()
                elif current_section:
                    # Append text to the current section
                    existing = extracted_sections[current_section]["content"]
                    extracted_sections[current_section]["content"] = f"{existing}\n{raw_text}".strip()

        doc.close()

        if doc_title:
            extracted_sections["title"]["content"] = doc_title
            extracted_sections["title"]["page"] = 1
        if doc_author:
            extracted_sections["authors"]["content"] = doc_author
            extracted_sections["authors"]["page"] = 1

        # Build normalized ResearchSectionSchema list
        result_sections: List[ResearchSectionSchema] = []
        for sec_id, default_title, default_content in CANONICAL_SECTIONS:
            item = extracted_sections.get(sec_id, {})
            content = item.get("content", "").strip()
            # If extracted content is empty, retain standard placeholder
            if not content:
                content = default_content
            
            result_sections.append(
                ResearchSectionSchema(
                    id=sec_id,
                    title=item.get("title", default_title),
                    content=content,
                    included=True,
                    pageNumber=item.get("page")
                )
            )

        metadata = {
            "title": doc_title or Path(file_path).stem.replace("_", " "),
            "authors": doc_author,
            "page_count": page_count
        }
        return metadata, result_sections

    @classmethod
    def parse_docx(cls, file_path: str) -> Tuple[Dict[str, Any], List[ResearchSectionSchema]]:
        doc = docx.Document(file_path)
        extracted_sections: Dict[str, Dict[str, Any]] = {}
        for sec_id, default_title, default_content in CANONICAL_SECTIONS:
            extracted_sections[sec_id] = {
                "title": default_title,
                "content": "",
                "page": 1
            }

        doc_title = ""
        current_section = None

        for para in doc.paragraphs:
            text = para.text.strip()
            if not text:
                continue

            if not doc_title and (para.style.name.startswith("Heading") or para.style.name == "Title" or len(text) < 150):
                doc_title = text
                extracted_sections["title"]["content"] = doc_title
                continue

            matched_sec = None
            for sec_key, pattern in SECTION_PATTERNS.items():
                if pattern.match(text):
                    matched_sec = sec_key
                    break
            
            if matched_sec:
                current_section = matched_sec
            elif current_section:
                existing = extracted_sections[current_section]["content"]
                extracted_sections[current_section]["content"] = f"{existing}\n{text}".strip()

        result_sections: List[ResearchSectionSchema] = []
        for sec_id, default_title, default_content in CANONICAL_SECTIONS:
            item = extracted_sections.get(sec_id, {})
            content = item.get("content", "").strip() or default_content
            result_sections.append(
                ResearchSectionSchema(
                    id=sec_id,
                    title=item.get("title", default_title),
                    content=content,
                    included=True,
                    pageNumber=1
                )
            )

        metadata = {
            "title": doc_title or Path(file_path).stem.replace("_", " "),
            "page_count": 1
        }
        return metadata, result_sections

    @classmethod
    def parse(cls, file_path: str, filename: str) -> Tuple[Dict[str, Any], List[ResearchSectionSchema]]:
        ext = Path(filename).suffix.lower()
        if ext == ".pdf":
            return cls.parse_pdf(file_path)
        elif ext in [".docx", ".doc"]:
            return cls.parse_docx(file_path)
        else:
            raise ValueError(f"Unsupported file format: {ext}")

document_parser_service = DocumentParserService()
