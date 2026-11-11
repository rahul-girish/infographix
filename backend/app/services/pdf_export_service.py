from __future__ import annotations

import base64
import io
from fastapi import HTTPException, status
from PIL import Image


def export_canvas_png_to_pdf(image_data_url: str) -> bytes:
    """Convert a high-resolution Konva PNG data URL into a single-page PDF byte stream."""
    if not image_data_url or "," not in image_data_url:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid image data URL provided for PDF export.",
        )

    try:
        _, encoded = image_data_url.split(",", 1)
        raw_bytes = base64.b64decode(encoded)
        with Image.open(io.BytesIO(raw_bytes)) as img:
            rgb_img = img.convert("RGB")
            pdf_buffer = io.BytesIO()
            rgb_img.save(pdf_buffer, format="PDF", resolution=150.0)
            return pdf_buffer.getvalue()
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to generate PDF from canvas image: {exc}",
        ) from exc
