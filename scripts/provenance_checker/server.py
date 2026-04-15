# ==============================================================================
# Copyright (c) 2026 Batangas State University (The National Engineering University)
# Project: IRIS Assistant System - Multimedia Provenance
# ==============================================================================

import asyncio
import json
import os
import tempfile
import time
import socket
import traceback
import shutil
from pathlib import Path
import threading
import signal

import yt_dlp
from fastapi import FastAPI, Query, UploadFile, File, HTTPException, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
from pydantic import BaseModel

from metadata_scanner import analyze_media, Decision
from vlm_classifier import VLMClassifier
from deepfake_detector import DeepfakeDetector

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

HTML_PATH = Path(__file__).parent / "index.html"
ONNX_PATH = "models/provenance/deepfake_detector.onnx"

local_ip = os.environ.get("LOCAL_IP", "localhost")
vlm_port = int(os.environ.get("VLM_PORT", 8001))
vlm_alias = os.environ.get("VLM_ALIAS", "gemma-4-e2b-it")

_vlm_client = OpenAI(base_url=f"http://{local_ip}:{vlm_port}/v1", api_key="EMPTY")
_detector: DeepfakeDetector | None = None

TOS_BLOCKED_DOMAINS = [
    "netflix.com", "hulu.com", "disneyplus.com", "primevideo.com", 
    "amazon.com", "max.com", "hbo.com", "onlyfans.com", "patreon.com"
]

PROVENANCE_TIMEOUT = int(os.environ.get("PROVENANCE_TIMEOUT", 0))
_last_activity_time = time.time()

@app.middleware("http")
async def update_activity_timer(request: Request, call_next):
    """Middleware that resets the idle timer on EVERY interaction."""
    global _last_activity_time
    _last_activity_time = time.time()
    response = await call_next(request)
    return response

@app.on_event("startup")
def inactivity_watcher():
    """Continuously monitors for inactivity and kills the server if abandoned."""
    if PROVENANCE_TIMEOUT > 0:
        def watcher():
            while True:
                time.sleep(10)  # Check the clock every 10 seconds
                idle_duration = time.time() - _last_activity_time
                
                if idle_duration > PROVENANCE_TIMEOUT:
                    print(f"\n[SECURITY] Provenance Server idle for over {PROVENANCE_TIMEOUT} seconds. Auto-closing to free GPU resources.\n")
                    os.kill(os.getpid(), signal.SIGTERM)
                    break
                    
        threading.Thread(target=watcher, daemon=True).start()

class ExplainRequest(BaseModel):
    prompt: str
    max_tokens: int = 450

@app.post("/explain")
async def explain(body: ExplainRequest):
    response = await asyncio.get_event_loop().run_in_executor(
        None,
        lambda: _vlm_client.chat.completions.create(
            model=vlm_alias,
            messages=[{"role": "user", "content": body.prompt}],
            max_tokens=body.max_tokens,
            temperature=0.3,
        )
    )
    return {"text": response.choices[0].message.content.strip()}

def get_detector() -> DeepfakeDetector:
    global _detector
    if _detector is None:
        _detector = DeepfakeDetector(onnx_path=ONNX_PATH)
    return _detector

@app.get("/", response_class=HTMLResponse)
async def serve_frontend(request: Request):
    client_ip = request.client.host
    print(f"\n👀 [ALERT] Provenance Kiosk accessed securely by IP: {client_ip}\n")
    
    return HTMLResponse(content=HTML_PATH.read_text(encoding="utf-8"))

@app.post("/upload")
async def upload_video(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.webm')):
        raise HTTPException(status_code=400, detail="Invalid video format.")
    tmp_dir = tempfile.mkdtemp()
    file_path = os.path.join(tmp_dir, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    return {"file_path": file_path}

def sse(event: str, data: dict) -> str: return f"event: {event}\ndata: {json.dumps(data)}\n\n"
def step_event(step: int, status: str, note: str = "") -> str: return sse("step", {"step": step, "status": status, "note": note})
def log_event(msg: str) -> str: return sse("log", {"message": msg})
def result_event(payload: dict) -> str: return sse("result", payload)
def error_event(msg: str) -> str: return sse("error", {"message": msg})

def download_video(url: str, out_dir: str) -> str:
    ydl_opts = {
        "format": "bestvideo[ext=mp4][height<=720]+bestaudio[ext=m4a]/best[ext=mp4]/best",
        "outtmpl": os.path.join(out_dir, "video.%(ext)s"),
        "quiet": True, "no_warnings": True, "merge_output_format": "mp4",
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl: ydl.download([url])
    for f in os.listdir(out_dir):
        if f.startswith("video."): return os.path.join(out_dir, f)
    raise FileNotFoundError("yt-dlp finished but no output file found.")

@app.get("/analyze")
async def analyze(url: str = Query(None), file_path: str = Query(None)):
    if not url and not file_path:
        return StreamingResponse(iter([error_event("Must provide URL or file.")]), media_type="text/event-stream")

    async def stream():
        start = time.perf_counter()
        tmp_dir = tempfile.mkdtemp()
        target_video_path = None

        try:
            yield log_event("Initializing IRIS Pipeline...")
            
            vlm_classifier = VLMClassifier(_vlm_client, vlm_alias)

            if url:
                if any(domain in url.lower() for domain in TOS_BLOCKED_DOMAINS):
                    yield error_event("TOS Violation: Prohibited domain."); return

                yield step_event(1, "active"); yield log_event(f"Target URL: {url}"); await asyncio.sleep(0.3); yield step_event(1, "complete")
                yield step_event(2, "active"); await asyncio.sleep(0.2); yield step_event(2, "complete")
                yield step_event(3, "active"); yield log_event("Downloading video via yt-dlp...")
                
                try: target_video_path = await asyncio.get_event_loop().run_in_executor(None, download_video, url, tmp_dir)
                except Exception as e: yield error_event(f"Download failed: {e}"); return
                
                yield step_event(3, "complete", note=os.path.basename(target_video_path))
                
            elif file_path:
                for i in range(1, 4): yield step_event(i, "skipped")
                yield log_event(f"Loaded local file: {os.path.basename(file_path)}")
                target_video_path = file_path

            # --- PROVENANCE CHECK ---
            yield step_event(4, "active")
            meta_result = await asyncio.get_event_loop().run_in_executor(None, analyze_media, target_video_path)
            yield step_event(4, "complete")

            if meta_result.isAIGenerated == Decision.YES:
                reason = str(meta_result.reason or "AI metadata confirmed.")
                yield log_event(f"METADATA HIT: {reason}. Bypassing downstream.")
                for i in range(5, 7): yield step_event(i, "skipped")
                yield result_event({
                    "verdict": "AI Generated", "verdictClass": "ai", "confidence": None, "videoType": "GENERATED", 
                    "description": reason, "elapsed": f"{time.perf_counter() - start:.2f}",
                    "trace": [
                        {"icon": "flag", "text": f"PROVENANCE — {reason}"},
                        {"icon": "done", "text": "FAST EXIT — AI origin confirmed via metadata."},
                        {"icon": "flag", "text": "CLASSIFIER — AI CONFIRMED."}
                    ],
                    "explanation": f"Metadata analysis identified AI generation: {reason}."
                })
                return

            # --- INTERLEAVED TEMPORAL PIPELINE ---
            yield log_event("No metadata signatures found. Entering Temporal Interleaved Analysis...")
            yield step_event(5, "active")
            yield step_event(6, "active")
            
            chunked_frames = await asyncio.get_event_loop().run_in_executor(
                None, vlm_classifier.get_chunked_video_frames, target_video_path, 2.0, 4
            )
            
            detector = await asyncio.get_event_loop().run_in_executor(None, get_detector)
            audio, vr, fps, tot_frames = await asyncio.get_event_loop().run_in_executor(
                None, detector.load_media, target_video_path
            )

            meta_reason = meta_result.reason if isinstance(meta_result.reason, str) else None
            final_trace = [{"icon": "done", "text": "PROVENANCE — No AI metadata found. Proceeding temporally."}]
            
            highest_fake_spike = 0.0
            onnx_evaluated_chunks = 0
            is_definitively_fake = False
            worst_chunk_frames = []
            
            for chunk in chunked_frames:
                c_idx, c_start, c_end = chunk['chunk_index'], chunk['start'], chunk['end']
                yield log_event(f"▶ Scanning Window {c_idx}/{len(chunked_frames)} [{c_start:.1f}s - {c_end:.1f}s]")
                
                # 1. Semantic VLM Gatekeeper
                v_verdict, v_reason = await asyncio.get_event_loop().run_in_executor(
                    None, vlm_classifier.classify_chunk, chunk['frames'], c_idx, c_start, c_end, meta_reason
                )

                print(v_reason)
                
                if v_verdict in ["ANIMATED", "REAL_NO_HUMANS"]:
                    yield log_event(f"  ↳ VLM: '{v_verdict}'. Skipping ONNX deepfake matrix.")
                    final_trace.append({"icon": "skip", "text": f"CHUNK {c_idx} [{c_start:.1f}s-{c_end:.1f}s] → VLM: {v_verdict}. Deepfake check bypassed."})
                    continue
                    
                # 2. Multimodal ONNX Execution
                yield log_event(f"  ↳ VLM: '{v_verdict}' (Human features detected). Engaging ONNX...")
                fake_prob = await asyncio.get_event_loop().run_in_executor(
                    None, detector.predict_window, audio, vr, fps, tot_frames, c_start, c_end
                )
                
                onnx_evaluated_chunks += 1
                if fake_prob > highest_fake_spike: 
                    highest_fake_spike = fake_prob
                    worst_chunk_frames = chunk['frames']
                if fake_prob > 0.5: 
                    is_definitively_fake = True

                    
                status = "FAKE" if fake_prob > 0.5 else "REAL"
                icon = "flag" if fake_prob > 0.5 else "done"
                yield log_event(f"  ↳ ONNX: {status} (Score: {fake_prob:.4f})")
                final_trace.append({"icon": icon, "text": f"CHUNK {c_idx} [{c_start:.1f}s-{c_end:.1f}s] → ONNX: {status} ({fake_prob:.3f})"})

            yield step_event(5, "complete")
            yield step_event(6, "complete")

            # --- AGGREGATE FINAL VERDICT ---
            if is_definitively_fake:
                final_verdict, v_class, final_type = "FAKE", "fake", "DEEPFAKE"
                desc = f"Deepfake artifacts detected during temporal analysis. Highest fake probability spike: {highest_fake_spike:.4f}."
                conf = int(highest_fake_spike * 100)
                flagged_frames_payload = worst_chunk_frames
            elif onnx_evaluated_chunks == 0:
                # VLM skipped every single chunk
                final_type = "ANIMATED / NO HUMANS"
                final_verdict = "SKIPPED"
                v_class = "skipped"
                desc = "No real human footage was detected in any window. The video was classified as animated or lacking humans."
                conf = None
                flagged_frames_payload = []
            else:
                final_verdict, v_class, final_type = "GENUINE", "genuine", "REAL"
                desc = f"No deepfake artifacts detected across {onnx_evaluated_chunks} human-verified window(s)."
                conf = int((1.0 - highest_fake_spike) * 100)
                flagged_frames_payload = []

            final_trace.append({"icon": "flag" if final_verdict in ["FAKE", "AI CONFIRMED"] else "done", "text": f"CLASSIFIER — Verdict: {final_verdict}."})

            yield result_event({
                "verdict": final_verdict, "verdictClass": v_class, "confidence": conf, "videoType": final_type,
                "description": desc, "trace": final_trace, "elapsed": f"{time.perf_counter() - start:.2f}",
                "explanation": f"The system processed the video in strict 2-second chronological windows. {desc}",
                "flagged_frames": flagged_frames_payload
            })

        except Exception as e:
            traceback.print_exc()
            yield error_event(f"Pipeline error: {e}")
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)
            if file_path and os.path.exists(os.path.dirname(file_path)):
                shutil.rmtree(os.path.dirname(file_path), ignore_errors=True)

    return StreamingResponse(stream(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})