"""
OCR Text Cleanup / Decryption

Uses pattern matching and optional AI to fix common OCR errors
in construction plan text.
"""

import re
import logging
import os
from typing import Optional, List, Tuple

logger = logging.getLogger(__name__)

# Common OCR errors in construction plans
# Format: (wrong_pattern, correct_text)
COMMON_OCR_FIXES = [
    # ============ Type/Inlet/Structure Names ============
    # Type variations
    (r'\bTIPE\b', 'TYPE'),
    (r'\bTYPF\b', 'TYPE'),
    (r'\bTYP[E3]\b', 'TYPE'),
    (r'\b7YPE\b', 'TYPE'),
    (r'\bTYPE\s+C\b', 'TYPE C'),
    (r'\bTYPE\s+D\b', 'TYPE D'),
    (r'\bTYPE\s+E\b', 'TYPE E'),
    (r'\bTYPE\s+P\b', 'TYPE P'),
    (r'\bTYPE\s+7\b', 'TYPE 7'),  # Manhole type
    
    # Inlet variations
    (r'\bINIET\b', 'INLET'),
    (r'\bINLFT\b', 'INLET'),
    (r'\bINL[E3]T\b', 'INLET'),
    (r'\bN[I1]LET\b', 'INLET'),
    (r'\bNLET\b', 'INLET'),
    (r'\b1NLET\b', 'INLET'),
    (r'\bINLE7\b', 'INLET'),
    (r'\bLNLET\b', 'INLET'),
    
    # Manhole variations
    (r'\bMANHOLE\b', 'MANHOLE'),
    (r'\bMANHOI[E3]\b', 'MANHOLE'),
    (r'\bMANH0LE\b', 'MANHOLE'),
    (r'\bMH\b', 'MH'),
    (r'\bM\.H\.\b', 'MH'),
    
    # Junction box
    (r'\bJUNCT[I1]ON\b', 'JUNCTION'),
    (r'\bJUNCTION\s+BOX\b', 'JUNCTION BOX'),
    (r'\bJ\.B\.\b', 'JB'),
    (r'\bJB\b', 'JB'),
    
    # ============ Pipe Types ============
    (r'\bROP\b', 'RCP'),  # Reinforced Concrete Pipe
    (r'\bR[C0]P\b', 'RCP'),
    (r'\bRGP\b', 'RCP'),
    (r'\bR\.C\.P\.\b', 'RCP'),
    (r'\bHDPE\b', 'HDPE'),
    (r'\bH[O0]PE\b', 'HDPE'),
    (r'\bHOPE\b', 'HDPE'),
    (r'\bH\.D\.P\.E\.\b', 'HDPE'),
    (r'\bPVC\b', 'PVC'),
    (r'\bPV[C0]\b', 'PVC'),
    (r'\bP\.V\.C\.\b', 'PVC'),
    (r'\bDIP\b', 'DIP'),  # Ductile Iron Pipe
    (r'\bD[I1]P\b', 'DIP'),
    (r'\bD\.I\.P\.\b', 'DIP'),
    (r'\bCMP\b', 'CMP'),  # Corrugated Metal Pipe
    (r'\bC\.M\.P\.\b', 'CMP'),
    (r'\bADS\b', 'ADS'),  # ADS drainage pipe
    (r'\bCPP\b', 'CPP'),  # Corrugated Plastic Pipe
    (r'\bPP\b', 'PP'),  # Plastic Pipe
    
    # ============ Structure Components ============
    (r'\bENDWALL\b', 'ENDWALL'),
    (r'\bENDWAIL\b', 'ENDWALL'),
    (r'\bENDWAL[I1L]\b', 'ENDWALL'),
    (r'\bEND\s*WALL\b', 'ENDWALL'),
    (r'\bE\.W\.\b', 'ENDWALL'),
    (r'\bMITERED\b', 'MITERED'),
    (r'\bMITURED\b', 'MITERED'),
    (r'\bM[I1]TERED\b', 'MITERED'),
    (r'\bM[I1]TRED\b', 'MITERED'),
    (r'\bSECTION\b', 'SECTION'),
    (r'\bSECT[I1]ON\b', 'SECTION'),
    (r'\bSECTIONF\b', 'SECTION'),
    (r'\bSEC7ION\b', 'SECTION'),
    (r'\bM[E3]S\b', 'MES'),  # Mitered End Section
    (r'\bFES\b', 'FES'),  # Flared End Section
    (r'\bF\.E\.S\.\b', 'FES'),
    (r'\bHEADWALL\b', 'HEADWALL'),
    (r'\bHEADWAIL\b', 'HEADWALL'),
    (r'\bAPRON\b', 'APRON'),
    (r'\bAPR0N\b', 'APRON'),
    (r'\bGRATE\b', 'GRATE'),
    (r'\bGRA7E\b', 'GRATE'),
    (r'\bFRAME\b', 'FRAME'),
    (r'\bCOVER\b', 'COVER'),
    
    # ============ Units ============
    (r'\bI[F1]\b', 'LF'),  # Linear Feet
    (r'\b[I1]F\b', 'LF'),
    (r'\bL\.F\.\b', 'LF'),
    (r'\bLIN\s*FT\b', 'LF'),
    (r'\bEA\b', 'EA'),  # Each
    (r'\bE[A4]\b', 'EA'),
    (r'\bSY\b', 'SY'),  # Square Yard
    (r'\bS\.Y\.\b', 'SY'),
    (r'\bCY\b', 'CY'),  # Cubic Yard
    (r'\bC\.Y\.\b', 'CY'),
    (r'\bSF\b', 'SF'),  # Square Feet
    (r'\bS\.F\.\b', 'SF'),
    (r'\bTON\b', 'TON'),
    (r'\bTONS\b', 'TON'),
    (r'\bLS\b', 'LS'),  # Lump Sum
    (r'\bL\.S\.\b', 'LS'),
    (r'\bGAL\b', 'GAL'),
    
    # ============ Size/Measurement Terms ============
    (r'\b(\d+)["\s]*[I1]NCH\b', r'\1 INCH'),
    (r'\b(\d+)["\s]*IN\b', r'\1 IN'),
    (r'\bD[I1]AMETER\b', 'DIAMETER'),
    (r'\bD[I1]A\b', 'DIA'),
    (r'\bRAD[I1]US\b', 'RADIUS'),
    
    # ============ Construction/Engineering Terms ============
    (r'\bPEOROS[E3]D\b', 'PROPOSED'),
    (r'\bPROPOS[E3]D\b', 'PROPOSED'),
    (r'\bPR0POSED\b', 'PROPOSED'),
    (r'\bUTUTY\b', 'UTILITY'),
    (r'\bUT[I1]L[I1]TY\b', 'UTILITY'),
    (r'\bUTILITIES\b', 'UTILITIES'),
    (r'\bEX[I1]ST[I1]NG\b', 'EXISTING'),
    (r'\bEXSOVENT\b', 'EXISTING'),
    (r'\bEXIST\b', 'EXIST'),
    (r'\bEX\.\b', 'EXIST'),
    (r'\bCONDU[I1]T\b', 'CONDUIT'),
    (r'\bC0NDUIT\b', 'CONDUIT'),
    (r'\bDRA[I1]NAGE\b', 'DRAINAGE'),
    (r'\bDRA1NAGE\b', 'DRAINAGE'),
    (r'\bSTRUCTUR[E3]\b', 'STRUCTURE'),
    (r'\bSTRUCTURES\b', 'STRUCTURES'),
    (r'\bSTR\b', 'STR'),
    (r'\bCONCR[E3]T[E3]\b', 'CONCRETE'),
    (r'\bCONC\b', 'CONC'),
    (r'\bR[E3][I1]NFORC[E3]D\b', 'REINFORCED'),
    (r'\bREINF\b', 'REINF'),
    (r'\bGALVAN[I1]Z[E3]D\b', 'GALVANIZED'),
    (r'\bGALV\b', 'GALV'),
    (r'\bST[E3][E3]L\b', 'STEEL'),
    (r'\bSTL\b', 'STL'),
    (r'\bALUM[I1]NUM\b', 'ALUMINUM'),
    (r'\bALUM\b', 'ALUM'),
    
    # ============ Directional/Location ============
    (r'\bENO\b', 'END'),
    (r'\bWINGED\b', 'WINGED'),
    (r'\bW[I1]NGED\b', 'WINGED'),
    (r'\bSTRA[I1]GHT\b', 'STRAIGHT'),
    (r'\bBARREL\b', 'BARREL'),
    (r'\bBARR[E3]L\b', 'BARREL'),
    (r'\bS[I1]NGLE\b', 'SINGLE'),
    (r'\bDOUBL[E3]\b', 'DOUBLE'),
    (r'\bTR[I1]PLE\b', 'TRIPLE'),
    (r'\bQUAD\b', 'QUAD'),
    (r'\bNORTH\b', 'NORTH'),
    (r'\bSOUTH\b', 'SOUTH'),
    (r'\bEAST\b', 'EAST'),
    (r'\bWEST\b', 'WEST'),
    (r'\bN\b(?=\s+\d)', 'N'),  # N direction
    (r'\bS\b(?=\s+\d)', 'S'),
    (r'\bE\b(?=\s+\d)', 'E'),
    (r'\bW\b(?=\s+\d)', 'W'),
    
    # ============ Elevations/Grades ============
    (r'\bCLASS\b', 'CLASS'),
    (r'\bC[L1]ASS\b', 'CLASS'),
    (r'\bGRADE\b', 'GRADE'),
    (r'\bGRAD[E3]\b', 'GRADE'),
    (r'\bELEV\b', 'ELEV'),
    (r'\bEL\.\b', 'EL'),
    (r'\bINV\b', 'INV'),  # Invert elevation
    (r'\b[I1]NV\b', 'INV'),
    (r'\bINVERT\b', 'INVERT'),
    (r'\bTC\b', 'TC'),   # Top of concrete
    (r'\bT\.C\.\b', 'TC'),
    (r'\bRIM\b', 'RIM'),
    (r'\bR[I1]M\b', 'RIM'),
    (r'\bFL\b', 'FL'),  # Flow line
    (r'\bF\.L\.\b', 'FL'),
    (r'\bSLOPE\b', 'SLOPE'),
    (r'\bSL0PE\b', 'SLOPE'),
    
    # ============ Station/Offset ============
    (r'\bSTATION\b', 'STATION'),
    (r'\bSTA\b', 'STA'),
    (r'\bSTA\.\b', 'STA'),
    (r'\bOFFSET\b', 'OFFSET'),
    (r'\b0FFSET\b', 'OFFSET'),
    (r'\bLT\b', 'LT'),  # Left
    (r'\bRT\b', 'RT'),  # Right
    (r'\bCL\b', 'CL'),  # Centerline
    (r'\bC\.L\.\b', 'CL'),
    (r'\bR/W\b', 'R/W'),  # Right of Way
    (r'\bROW\b', 'ROW'),
    
    # ============ FDOT/Pay Item Related ============
    (r'\bFDOT\b', 'FDOT'),
    (r'\bFD0T\b', 'FDOT'),
    (r'\bPAY\s*[I1]TEM\b', 'PAY ITEM'),
    (r'\bQTY\b', 'QTY'),
    (r'\bQUANT[I1]TY\b', 'QUANTITY'),
    (r'\bUN[I1]T\b', 'UNIT'),
    (r'\bDESCR[I1]PT[I1]ON\b', 'DESCRIPTION'),
    (r'\bDESC\b', 'DESC'),
    (r'\bSPEC\b', 'SPEC'),
    (r'\bSPEC[I1]F[I1]CAT[I1]ON\b', 'SPECIFICATION'),
    
    # ============ Sheet/Plan Terms ============
    (r'\bSHEET\b', 'SHEET'),
    (r'\bSH[E3]ET\b', 'SHEET'),
    (r'\bPLAN\b', 'PLAN'),
    (r'\bPROF[I1]LE\b', 'PROFILE'),
    (r'\bDETAIL\b', 'DETAIL'),
    (r'\bD[E3]TAIL\b', 'DETAIL'),
    (r'\bNOTES\b', 'NOTES'),
    (r'\bLEGEND\b', 'LEGEND'),
    (r'\bSCALE\b', 'SCALE'),
    (r'\bSCAL[E3]\b', 'SCALE'),
    
    # ============ Common Number/Letter Swaps ============
    (r'\b0\b(?=[A-Z])', 'O'),  # 0 before letter -> O
    (r'(?<=[A-Z])0(?=[A-Z])', 'O'),  # 0 between letters -> O
    (r'\b1\b(?=[A-Z])', 'I'),  # 1 before letter -> I
    (r'(?<=[A-Z])1(?=[A-Z])', 'I'),  # 1 between letters -> I
]

# Size patterns to normalize
SIZE_PATTERNS = [
    # "24 RCP" or "24" RCP" -> "24" RCP"
    (r'(\d{1,2})\s*["\u201d\u2033]?\s*(RCP|HDPE|PVC|DIP|PIPE)', r'\1" \2'),
    # "24-INCH" -> "24""
    (r'(\d{1,2})\s*-?\s*INCH', r'\1"'),
    # "24 IN" -> "24""
    (r'(\d{1,2})\s+IN\b', r'\1"'),
]


def apply_regex_fixes(text: str) -> str:
    """Apply regex-based OCR error corrections."""
    result = text.upper()  # Normalize to uppercase for matching
    
    for pattern, replacement in COMMON_OCR_FIXES:
        result = re.sub(pattern, replacement, result, flags=re.IGNORECASE)
    
    for pattern, replacement in SIZE_PATTERNS:
        result = re.sub(pattern, replacement, result, flags=re.IGNORECASE)
    
    return result


def cleanup_ocr_text(text: str, use_ai: bool = False, api_key: Optional[str] = None) -> str:
    """
    Clean up OCR-extracted text from construction plans.
    
    Args:
        text: Raw OCR text
        use_ai: Whether to use AI for additional cleanup
        api_key: Anthropic API key (required if use_ai=True)
    
    Returns:
        Cleaned text with common OCR errors fixed
    """
    if not text:
        return text
    
    # Step 1: Apply regex fixes
    cleaned = apply_regex_fixes(text)
    
    # Count fixes made
    original_upper = text.upper()
    if cleaned != original_upper:
        # Rough count of changes
        changes = sum(1 for a, b in zip(original_upper.split(), cleaned.split()) if a != b)
        logger.info(f"OCR cleanup: ~{changes} corrections applied via regex")
    
    # Step 2: Optional AI cleanup for complex errors
    if use_ai and api_key:
        cleaned = _ai_cleanup(cleaned, api_key)
    
    return cleaned


def _ai_cleanup(text: str, api_key: str) -> str:
    """Use AI to fix complex OCR errors that regex can't handle."""
    try:
        import anthropic
        
        client = anthropic.Anthropic(api_key=api_key)
        
        # Only send a sample if text is very long
        sample = text[:3000] if len(text) > 3000 else text
        
        prompt = f"""You are an OCR error correction specialist for construction plans.

The following text was extracted via OCR from a Florida construction/drainage plan. 
Fix obvious OCR errors while preserving the structure. Focus on:
- Construction terms (INLET, MANHOLE, RCP, HDPE, ENDWALL, MES, etc.)
- Pipe sizes (12", 18", 24", 30", etc.)
- Pay item numbers (format: XXX-XXX-XXX)
- Quantities and units (LF, EA, SY, CY)

Only fix clear OCR errors. Don't add information that isn't there.
Return ONLY the corrected text, no explanations.

Text to correct:
{sample}"""

        response = client.messages.create(
            model="claude-3-5-haiku-20241022",
            max_tokens=4000,
            messages=[{"role": "user", "content": prompt}]
        )
        
        corrected = response.content[0].text
        logger.info(f"AI OCR cleanup applied ({len(sample)} chars processed)")
        
        # If we only processed a sample, append the rest
        if len(text) > 3000:
            corrected += text[3000:]
        
        return corrected
        
    except Exception as e:
        logger.warning(f"AI OCR cleanup failed: {e}")
        return text


def extract_materials_from_messy_text(text: str) -> List[dict]:
    """
    Extract material mentions from messy OCR text using flexible patterns.
    
    This is a fallback for when structured parsing fails.
    Returns a list of potential materials with low confidence.
    """
    materials = []
    
    # Clean the text first
    cleaned = apply_regex_fixes(text)
    
    # Look for pipe callouts (e.g., "24" RCP", "18 INCH PIPE")
    pipe_pattern = r'(\d{1,2})["\u201d\u2033]?\s*(RCP|HDPE|PVC|DIP|PIPE|CONC)'
    for match in re.finditer(pipe_pattern, cleaned, re.IGNORECASE):
        size = match.group(1)
        pipe_type = match.group(2).upper()
        if pipe_type == 'CONC':
            pipe_type = 'RCP'  # Concrete pipe = RCP
        materials.append({
            'type': 'pipe',
            'size': f'{size}"',
            'material': pipe_type,
            'raw_match': match.group(0),
            'confidence': 'low'
        })
    
    # Look for inlet/manhole mentions
    structure_pattern = r'(TYPE\s+[A-Z])?\s*(INLET|MANHOLE|MH|JUNCTION BOX)\s*#?\s*(\d+)?'
    for match in re.finditer(structure_pattern, cleaned, re.IGNORECASE):
        struct_type = match.group(2).upper()
        if struct_type == 'MH':
            struct_type = 'MANHOLE'
        inlet_type = match.group(1) if match.group(1) else ''
        inlet_id = match.group(3) if match.group(3) else ''
        materials.append({
            'type': struct_type.lower(),
            'subtype': inlet_type.strip() if inlet_type else None,
            'id': inlet_id if inlet_id else None,
            'raw_match': match.group(0),
            'confidence': 'low'
        })
    
    # Look for endwall/MES mentions
    endwall_pattern = r'(STRAIGHT|WINGED|U-TYPE)?\s*(ENDWALL|MES|MITERED END SECTION|FES|FLARED END)'
    for match in re.finditer(endwall_pattern, cleaned, re.IGNORECASE):
        end_type = match.group(2).upper()
        if 'MES' in end_type or 'MITERED' in end_type:
            end_type = 'MES'
        elif 'FES' in end_type or 'FLARED' in end_type:
            end_type = 'FES'
        else:
            end_type = 'ENDWALL'
        style = match.group(1).upper() if match.group(1) else 'STANDARD'
        materials.append({
            'type': end_type.lower(),
            'style': style,
            'raw_match': match.group(0),
            'confidence': 'low'
        })
    
    # Deduplicate
    seen = set()
    unique_materials = []
    for m in materials:
        key = (m.get('type'), m.get('size'), m.get('material'), m.get('id'))
        if key not in seen:
            seen.add(key)
            unique_materials.append(m)
    
    logger.info(f"Extracted {len(unique_materials)} potential materials from messy text")
    return unique_materials


if __name__ == "__main__":
    # Test with sample OCR text
    test_text = """
    TIPE C INLET 30.00 PEOROSED UTUTY EXSOVENT ROP 4° HOPE CONDUIT FOR TIPE C NLET
    MITURED ENO SECTIONF 24" ROP INV 27.25
    TYPE E INLET 622 TC 32.50 RCP 27.25
    """
    
    print("Original:")
    print(test_text)
    print("\nCleaned:")
    print(cleanup_ocr_text(test_text))
    print("\nExtracted materials:")
    for m in extract_materials_from_messy_text(test_text):
        print(f"  {m}")
