# utils.py
from __future__ import annotations
import os
from typing import List, Tuple, Optional
from datetime import datetime

# DOCX
from docx import Document
from docx.shared import Pt, Inches
from docx.enum.text import WD_PARAGRAPH_ALIGNMENT
from docx.oxml.ns import qn
from docx.enum.style import WD_STYLE_TYPE

# PDF (ReportLab)
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, ListFlowable, ListItem, Image, Table, TableStyle
)
from reportlab.lib.enums import TA_LEFT, TA_CENTER
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.rl_config import defaultPageSize

# Pillow for image reading (logo)
from PIL import Image as PILImage

# -------------------------
# Config
# -------------------------
OUT_DIR = "/mnt/data" if os.path.exists("/mnt/data") else os.getcwd()
if not os.path.exists(OUT_DIR):
    os.makedirs(OUT_DIR, exist_ok=True)

HEADING_COLOR_HEX = "#1F4E79"  # deep blue for headings
HEADING_COLOR = colors.HexColor(HEADING_COLOR_HEX)
BODY_FONT = "Helvetica"  # fallback
BODY_FONT_BOLD = "Helvetica-Bold"

# Optionally register a nicer TTF if available. Uncomment and provide path if you have a preferred font file.
# Example:
# pdfmetrics.registerFont(TTFont('Inter', '/usr/share/fonts/truetype/inter/Inter-Regular.ttf'))
# BODY_FONT = "Inter"
# BODY_FONT_BOLD = "Inter-Bold"

KNOWN_SECTIONS = {
    "profile summary", "core skills", "core competencies", "technical skills",
    "work experience", "education", "certifications", "projects", "awards", "skills"
}

# -------------------------
# Helpers (text parsing)
# -------------------------
def _normalize_line(s: str) -> str:
    return s.strip()

def _parse_lines(text: str) -> List[Tuple[str, str]]:
    """
    Parse plain text into typed lines:
    returns list of (type, content)
    types: heading, bullet, para, hr, blank
    Heading is detected if line matches known heading (case-insensitive).
    """
    out = []
    for raw in (text or "").splitlines():
        ln = raw.rstrip()
        if not ln:
            out.append(("blank", ""))
            continue
        s = ln.strip()
        low = s.lower().rstrip(":")
        if low in KNOWN_SECTIONS:
            out.append(("heading", s.title()))
        elif s.startswith("• ") or s.startswith("- ") or s.startswith("* "):
            out.append(("bullet", s[2:].strip()))
        elif s.startswith("---") or s.startswith("___"):
            out.append(("hr", ""))
        else:
            out.append(("para", s))
    return out

def _extract_title_and_contact(text: str) -> Tuple[str, str]:
    """
    Heuristic: first non-empty line short -> title (name).
    Next 1-3 lines: pick one with @ or digits or http -> contact
    """
    lines = [ln.strip() for ln in (text or "").splitlines() if ln.strip()]
    title = ""
    contact = ""
    if lines:
        first = lines[0]
        if 1 <= len(first.split()) <= 5 and len(first) <= 60:
            title = first
            rest = lines[1:6]
        else:
            rest = lines[:6]
    else:
        rest = []
    # find contact-like line
    for ln in rest:
        if "@" in ln or "http" in ln or any(ch.isdigit() for ch in ln):
            contact = ln
            break
    return title, contact

def _safe_filepath(basename: str, ext: str) -> str:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = (basename or "document").strip().replace(" ", "_")
    filename = f"{base}_{stamp}.{ext.lstrip('.')}"
    return os.path.join(OUT_DIR, filename)

# -------------------------
# DOCX export (polished)
# -------------------------
# add imports at top of file if not already:
from io import BytesIO

# Replace or add these functions in utils.py

def export_docx(text: str, out_path: Optional[str] = None, logo_path: Optional[str] = None):
    """
    Export DOCX. If out_path is None -> return bytes (in-memory).
    If out_path provided -> write file and return out_path.
    """
    # create Document
    doc = Document()
    # set page margins
    for sec in doc.sections:
        sec.top_margin = Inches(0.6)
        sec.bottom_margin = Inches(0.6)
        sec.left_margin = Inches(0.7)
        sec.right_margin = Inches(0.7)

    # Styles
    styles = doc.styles
    normal = styles["Normal"]
    normal.font.name = "Calibri"
    try:
        normal._element.rPr.rFonts.set(qn('w:eastAsia'), 'Calibri')
    except Exception:
        pass
    normal.font.size = Pt(11)

    # Heading style
    if "TailorHeading" not in [s.name for s in styles]:
        hstyle = styles.add_style("TailorHeading", WD_STYLE_TYPE.PARAGRAPH)
        hstyle.font.name = "Calibri"
        hstyle.font.size = Pt(13)
        hstyle.font.bold = True
        try:
            from docx.shared import RGBColor
            hstyle.font.color.rgb = RGBColor(0x1F, 0x4E, 0x79)
        except Exception:
            pass
    else:
        hstyle = styles["TailorHeading"]

    # Title / Header block (use heuristics)
    title, contact = _extract_title_and_contact(text)
    if title:
        p = doc.add_paragraph()
        run = p.add_run(title.strip())
        run.bold = True
        run.font.size = Pt(18)
        p.alignment = WD_PARAGRAPH_ALIGNMENT.CENTER
        if contact:
            p2 = doc.add_paragraph(contact.strip())
            p2.alignment = WD_PARAGRAPH_ALIGNMENT.CENTER
            p2.runs[0].font.size = Pt(9)
            p2.runs[0].italic = True
        doc.add_paragraph()

    # optional logo in header (preserve if provided)
    if logo_path and os.path.exists(logo_path):
        try:
            header = doc.sections[0].header
            ph = header.paragraphs[0]
            run = ph.add_run()
            run.add_picture(logo_path, width=Inches(1.0))
            ph.alignment = WD_PARAGRAPH_ALIGNMENT.LEFT
        except Exception:
            pass

    # Walk parsed lines and emit docx content
    parsed = _parse_lines(text)
    start_idx = 0
    while start_idx < len(parsed) and parsed[start_idx][0] == "blank":
        start_idx += 1
    if title and start_idx < len(parsed):
        ttype, tcontent = parsed[start_idx]
        if tcontent.strip() == title.strip():
            start_idx += 1
            if start_idx < len(parsed) and parsed[start_idx][1].strip() == contact.strip():
                start_idx += 1

    for t, content in parsed[start_idx:]:
        if t == "heading":
            p = doc.add_paragraph()
            p.style = hstyle
            p.add_run(content)
        elif t == "bullet":
            doc.add_paragraph(content, style="List Bullet")
        elif t == "hr":
            doc.add_paragraph("_" * 60)
        elif t == "para":
            doc.add_paragraph(content)
        else:
            doc.add_paragraph("")

    # If out_path provided -> save to disk; else return bytes
    if out_path:
        doc.save(out_path)
        return out_path

    bio = BytesIO()
    doc.save(bio)
    bio.seek(0)
    return bio.getvalue()


def export_pdf(text: str, out_path: Optional[str] = None, logo_path: Optional[str] = None):
    """
    Export PDF. If out_path is None -> return bytes (in-memory).
    If out_path provided -> write file and return out_path.
    """
    PAGE_WIDTH, PAGE_HEIGHT = A4
    margin = 18 * mm
    usable_width = PAGE_WIDTH - 2 * margin

    stylesheet = getSampleStyleSheet()
    body_style = ParagraphStyle(
        "BodyX",
        parent=stylesheet["Normal"],
        fontName=BODY_FONT,
        fontSize=10.5,
        leading=12,
        spaceAfter=4,
        alignment=TA_LEFT
    )
    heading_style = ParagraphStyle(
        "HeadingX",
        parent=stylesheet["Heading2"],
        fontName=BODY_FONT_BOLD,
        fontSize=12.5,
        leading=14,
        textColor=HEADING_COLOR,
        spaceBefore=6,
        spaceAfter=4,
        alignment=TA_LEFT
    )

    story = []

    title, contact = _extract_title_and_contact(text)

    if logo_path and os.path.exists(logo_path):
        try:
            pil = PILImage.open(logo_path)
            w, h = pil.size
            target_h = 48
            scale = target_h / float(h)
            logo_w = int(w * scale)
            img = Image(logo_path, width=logo_w, height=target_h)
            right = []
            if title:
                right.append(Paragraph(f"<b>{title}</b>", ParagraphStyle("Htitle", parent=heading_style, fontSize=16)))
            if contact:
                right.append(Paragraph(contact, ParagraphStyle("ContactSmall", parent=body_style, fontSize=9)))
            tbl = Table([[img, right]], colWidths=[logo_w + 6, usable_width - (logo_w + 6)])
            tbl.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                                     ("LEFTPADDING", (0,0), (-1,-1), 0),
                                     ("RIGHTPADDING", (0,0), (-1,-1), 0),
                                     ("TOPPADDING", (0,0), (-1,-1), 0),
                                     ("BOTTOMPADDING", (0,0), (-1,-1), 0)]))
            story.append(tbl)
            story.append(Spacer(1, 6))
        except Exception:
            if title:
                story.append(Paragraph(f"<b>{title}</b>", ParagraphStyle("Htitle", parent=heading_style, fontSize=16)))
            if contact:
                story.append(Paragraph(contact, ParagraphStyle("ContactSmall", parent=body_style, fontSize=9)))
            story.append(Spacer(1, 6))
    else:
        if title:
            story.append(Paragraph(f"<b>{title}</b>", ParagraphStyle("Htitle", parent=heading_style, fontSize=16)))
        if contact:
            story.append(Paragraph(contact, ParagraphStyle("ContactSmall", parent=body_style, fontSize=9)))
        story.append(Spacer(1, 6))

    parsed = _parse_lines(text)

    p_idx = 0
    while p_idx < len(parsed) and parsed[p_idx][0] == "blank":
        p_idx += 1
    if title and p_idx < len(parsed):
        ttype, tcontent = parsed[p_idx]
        if tcontent.strip() == title.strip():
            p_idx += 1
            if p_idx < len(parsed) and parsed[p_idx][1].strip() == contact.strip():
                p_idx += 1

    bullets_buf = []

    def flush_bullets_to_story():
        nonlocal bullets_buf, story
        if not bullets_buf:
            return
        items = [ListItem(Paragraph(b, body_style), leftIndent=6) for b in bullets_buf]
        lf = ListFlowable(items, bulletType="bullet", start="disc", leftIndent=12, bulletFontName=BODY_FONT)
        story.append(lf)
        story.append(Spacer(1, 4))
        bullets_buf = []

    for typ, content in parsed[p_idx:]:
        if typ == "heading":
            flush_bullets_to_story()
            box = Table([[Paragraph(f"<b>{content}</b>", heading_style)]], colWidths=[usable_width])
            box.setStyle(TableStyle([
                ("BACKGROUND", (0,0), (-1,-1), colors.whitesmoke),
                ("LEFTPADDING", (0,0), (-1,-1), 6),
                ("RIGHTPADDING", (0,0), (-1,-1), 6),
                ("TOPPADDING", (0,0), (-1,-1), 4),
                ("BOTTOMPADDING", (0,0), (-1,-1), 4),
                ("BOX", (0,0), (-1,-1), 0.4, colors.HexColor("#EEEEEE"))
            ]))
            story.append(box)
            story.append(Spacer(1, 6))
        elif typ == "bullet":
            bullets_buf.append(content)
        elif typ == "para":
            flush_bullets_to_story()
            story.append(Paragraph(content, body_style))
            story.append(Spacer(1, 4))
        elif typ == "hr":
            flush_bullets_to_story()
            hr = Table([[""]], colWidths=[usable_width])
            hr.setStyle(TableStyle([("LINEBELOW", (0,0), (-1,-1), 0.4, colors.HexColor("#DDDDDD"))]))
            story.append(hr)
            story.append(Spacer(1, 4))
        else:
            flush_bullets_to_story()
            story.append(Spacer(1, 4))
    flush_bullets_to_story()

    # Build into BytesIO if out_path is None
    if out_path:
        doc = SimpleDocTemplate(out_path, pagesize=A4,
                                leftMargin=margin, rightMargin=margin,
                                topMargin=margin, bottomMargin=margin)
        doc.build(story)
        return out_path

    bio = BytesIO()
    doc = SimpleDocTemplate(bio, pagesize=A4,
                            leftMargin=margin, rightMargin=margin,
                            topMargin=margin, bottomMargin=margin)
    doc.build(story)
    bio.seek(0)
    return bio.getvalue()