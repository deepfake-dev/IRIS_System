import c2pa
import json
import os
import re
import xml.etree.ElementTree as ET
from enum import Enum
from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple, List, Union

from hachoir.parser import createParser
from hachoir.metadata import extractMetadata

# ── Configuration & Constants ─────────────────────────────────────────────────

class Decision(Enum):
    YES = "YES"
    MAYBE = "MAYBE"

@dataclass
class AnalysisResult:
    isAIGenerated: Decision
    reason: Optional[Union[str, Dict[str, Any]]] = None

XMP_NS = {
    "rdf":   "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
    "xmp":   "http://ns.adobe.com/xap/1.0/",
    "xmpMM": "http://ns.adobe.com/xap/1.0/mm/",
    "stEvt": "http://ns.adobe.com/xap/1.0/sType/ResourceEvent#",
    "dc":    "http://purl.org/dc/elements/1.1/",
    "photoshop": "http://ns.adobe.com/photoshop/1.0/",
    "xmpRights": "http://ns.adobe.com/xap/1.0/rights/",
    "c2pa":  "http://c2pa.org/manifest",
}

# 1. Terms that guarantee AI Generation
SURE_AI_KEYWORDS = [
    "sora", "dall-e", "dalle", "midjourney", "stable diffusion",
    "firefly", "generative ai", "ai generated", "artificial intelligence",
    "openai", "google veo", "veo", "imagen", "runway", "pika",
    "kling", "hailuo", "luma", "dreamachine", "invideo", "synthesia",
    "heygen", "d-id", "deepfake"
]

SURE_C2PA_SIGNALS = [
    "trainedAlgorithmicMedia", "compositeWithTrainedAlgorithmicMedia",
    "openai", "google", "adobe firefly", "stability ai",
    "midjourney", "runway", "pika", "generative ai"
]

# 2. Terms that imply manipulation/editing, but don't guarantee full AI generation
SUSPICIOUS_KEYWORDS = [
    "composite", "synthetic", "rendered", "photoshop", "premiere", 
    "after effects", "capcut", "davinci resolve", "edited", "modified", 
    "lightroom", "final cut"
]

# ── Helper Functions ──────────────────────────────────────────────────────────

def get_matched_keyword(text: str, keyword_list: List[str]) -> Optional[str]:
    """Returns the specific keyword found, or None if no match."""
    if not text:
        return None
    text_lower = str(text).lower()
    for keyword in keyword_list:
        if keyword in text_lower:
            return keyword
    return None

def extract_xmp_bytes(file_path: str) -> Optional[str]:
    try:
        with open(file_path, "rb") as f:
            data = f.read()
        start = data.find(b"<?xpacket begin")
        if start == -1:
            start = data.find(b"<x:xmpmeta")
        if start == -1:
            return None

        end = data.find(b"<?xpacket end", start)
        if end != -1:
            end = data.find(b">", end) + 1
        else:
            end = data.find(b"</x:xmpmeta>", start)
            if end != -1:
                end += len(b"</x:xmpmeta>")
                
        if end == -1:
            return None
        return data[start:end].decode("utf-8", errors="replace")
    except Exception:
        return None

# ── Core Scanning Functions ───────────────────────────────────────────────────

def scan_c2pa(file_path: str) -> Tuple[bool, Optional[str], Optional[str], Dict[str, Any]]:
    """Returns: (has_c2pa, sure_ai_tool, suspicious_term, manifest_data)"""
    try:
        reader = c2pa.Reader(file_path)
        manifest_json = json.loads(reader.json())
        active_manifest_id = manifest_json.get("active_manifest")
        
        if not active_manifest_id:
            return False, None, None, {}

        manifest_data = manifest_json.get("manifests", {}).get(active_manifest_id, {})
        sure_tool = None
        sus_tool = None
        
        # Check Claim Generator
        for gen_info in manifest_data.get("claim_generator_info", []):
            name = gen_info.get("name", "")
            if match := get_matched_keyword(name, SURE_C2PA_SIGNALS): sure_tool = match
            if match := get_matched_keyword(name, SUSPICIOUS_KEYWORDS): sus_tool = match
                
        # Check Signature Issuer
        sig = manifest_data.get("signature_info", {})
        for field in [sig.get("issuer", ""), sig.get("common_name", "")]:
            if match := get_matched_keyword(field, SURE_C2PA_SIGNALS): sure_tool = match
            if match := get_matched_keyword(field, SUSPICIOUS_KEYWORDS): sus_tool = match

        # Check Assertions
        for assertion in manifest_data.get("assertions", []):
            if assertion.get("label") in ("c2pa.actions", "c2pa.actions.v2"):
                for action in assertion.get("data", {}).get("actions", []):
                    for field in [action.get("softwareAgent", ""), action.get("digitalSourceType", "")]:
                        if match := get_matched_keyword(field, SURE_C2PA_SIGNALS): sure_tool = match
                        if match := get_matched_keyword(field, SUSPICIOUS_KEYWORDS): sus_tool = match

        return True, sure_tool, sus_tool, manifest_data
    except Exception:
        return False, None, None, {}


def scan_xmp(file_path: str) -> Tuple[bool, Optional[str], Optional[str], Dict[str, Any]]:
    """Returns: (has_xmp, sure_ai_tool, suspicious_term, extracted_xmp_fields)"""
    xmp_str = extract_xmp_bytes(file_path)
    if not xmp_str:
        return False, None, None, {}

    try:
        clean_xml = re.sub(r'<\?xpacket[^?]*\?>', '', xmp_str).strip()
        root = ET.fromstring(clean_xml)
    except Exception:
        return False, None, None, {}

    extracted_data = {}
    sure_tool = None
    sus_tool = None

    for label, tag in {"CreatorTool": f"{{{XMP_NS['xmp']}}}CreatorTool", 
                       "Description": f"{{{XMP_NS['dc']}}}description"}.items():
        for elem in root.iter(tag):
            val = (elem.text or "".join(elem.itertext())).strip()
            if val:
                extracted_data[label] = val
                if match := get_matched_keyword(val, SURE_AI_KEYWORDS): sure_tool = match
                if match := get_matched_keyword(val, SUSPICIOUS_KEYWORDS): sus_tool = match

    if raw_match := get_matched_keyword(xmp_str, SURE_AI_KEYWORDS):
        if not sure_tool: sure_tool = raw_match
    elif raw_sus := get_matched_keyword(xmp_str, SUSPICIOUS_KEYWORDS):
        if not sus_tool: sus_tool = raw_sus
        
    extracted_data["raw_xmp_length"] = len(xmp_str)
    return True, sure_tool, sus_tool, extracted_data


def scan_basic_metadata(file_path: str) -> Tuple[bool, Optional[str], Optional[str], Dict[str, str]]:
    """Returns: (has_meta, sure_ai_tool, suspicious_term, extracted_metadata)"""
    try:
        parser = createParser(file_path)
        if not parser:
            return False, None, None, {}
        with parser:
            metadata = extractMetadata(parser)
        if not metadata:
            return False, None, None, {}
            
        extracted_data = {}
        sure_tool = None
        sus_tool = None
        
        for item in metadata.exportPlaintext():
            parts = item.split(":", 1)
            if len(parts) == 2:
                key = parts[0].strip("- ")
                val = parts[1].strip()
                extracted_data[key] = val
                if match := get_matched_keyword(val, SURE_AI_KEYWORDS): sure_tool = match
                if match := get_matched_keyword(val, SUSPICIOUS_KEYWORDS): sus_tool = match
                    
        return True, sure_tool, sus_tool, extracted_data
    except Exception:
        return False, None, None, {}

# ── The Main API Endpoint Function ────────────────────────────────────────────

def analyze_media(file_path: str) -> AnalysisResult:
    """
    Evaluates a media file and returns a structured decision matrix.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    # 1. Evaluate C2PA 
    has_c2pa, c2pa_sure, c2pa_sus, c2pa_manifest = scan_c2pa(file_path)
    
    if has_c2pa:
        if c2pa_sure:
            return AnalysisResult(isAIGenerated=Decision.YES, reason=f"Indications of Generation by {c2pa_sure.title()}")
        elif c2pa_sus:
            return AnalysisResult(isAIGenerated=Decision.MAYBE, reason={"c2pa_manifest": c2pa_manifest})
        else:
            return AnalysisResult(isAIGenerated=Decision.NO, reason=None)

    # 2. Evaluate EXIF/XMP/Metadata
    has_xmp, xmp_sure, xmp_sus, xmp_data = scan_xmp(file_path)
    has_meta, meta_sure, meta_sus, meta_data = scan_basic_metadata(file_path)
    
    has_any_meta = has_xmp or has_meta
    sure_tool = xmp_sure or meta_sure
    sus_tool = xmp_sus or meta_sus

    if has_any_meta:
        if sure_tool:
            return AnalysisResult(isAIGenerated=Decision.YES, reason=f"Indications of Generation by {sure_tool.title()}")
        elif sus_tool:
            combined_reason = {}
            if xmp_data: combined_reason["xmp_metadata"] = xmp_data
            if meta_data: combined_reason["basic_metadata"] = meta_data
            return AnalysisResult(isAIGenerated=Decision.MAYBE, reason=combined_reason)
        else:
            # File has metadata, but no AI or suspicious terms were found
            return AnalysisResult(isAIGenerated=Decision.MAYBE, reason="Cannot find metadata-level anomalies. Have it checked by the VLM!")

    # 3. No C2PA and No Metadata
    return AnalysisResult(isAIGenerated=Decision.MAYBE, reason="Cannot find metadata-level anomalies. Have it checked by the VLM!")