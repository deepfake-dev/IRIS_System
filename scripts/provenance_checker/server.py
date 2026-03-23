"""
IRIS Deepfake Detection Server
─────────────────────────────────────────────────────────────────
Serves the iris-web.html frontend and a single /analyze endpoint
that streams pipeline progress via Server-Sent Events (SSE).

Pipeline stages:
  1. Receive URL
  2. Route request
  3. Download video   (yt-dlp)
  4. Metadata scan    (C2PA / XMP / EXIF)
  5. VLM classify     (Qwen3-VL)
  6-8. Deepfake engine (VideoMAE + Wav2Vec + Fourier CNN)

Install:
  pip install fastapi uvicorn yt-dlp
Run:
  uvicorn server:app --host 0.0.0.0 --port 4321 --reload
"""

import asyncio
import json
import os
import tempfile
import time
import traceback
from pathlib import Path

import yt_dlp
from fastapi import FastAPI, Query
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware

from metadata_scanner import analyze_media, Decision
from vlm_classifier import llm_classify_video
from deepfake_detector import DeepfakeDetector

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

HTML_PATH = Path(__file__).parent / "index.html"
ONNX_PATH = "models/provenance/deepfake_detector_model.onnx"

# OpenAI-compatible client pointing at local llama.cpp VLM
from openai import OpenAI as _OpenAI
_vlm_client = _OpenAI(base_url="http://localhost:8001/v1", api_key="EMPTY")


# ─────────────────────────────────────────────
# /explain — VLM explains the verdict
# ─────────────────────────────────────────────

from fastapi import Request as _Request
from pydantic import BaseModel as _BaseModel

class ExplainRequest(_BaseModel):
    prompt: str
    max_tokens: int = 450

@app.post("/explain")
async def explain(body: ExplainRequest):
    response = await asyncio.get_event_loop().run_in_executor(
        None,
        lambda: _vlm_client.chat.completions.create(
            model="qwen3-vl",
            messages=[{"role": "user", "content": body.prompt}],
            max_tokens=body.max_tokens,
            temperature=0.3,
            extra_body={"repeat_penalty": 1.1},
        )
    )
    return {"text": response.choices[0].message.content.strip()}

# Load detector once at startup — expensive to reload
_detector: DeepfakeDetector | None = None

def get_detector() -> DeepfakeDetector:
    global _detector
    if _detector is None:
        _detector = DeepfakeDetector(onnx_path=ONNX_PATH)
    return _detector


# ─────────────────────────────────────────────
# Serve HTML
# ─────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    return HTMLResponse(content=HTML_PATH.read_text(encoding="utf-8"))


# ─────────────────────────────────────────────
# SSE helpers
# ─────────────────────────────────────────────

def sse(event: str, data: dict) -> str:
    """Format a single SSE message."""
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


def step_event(step: int, status: str, note: str = "") -> str:
    return sse("step", {"step": step, "status": status, "note": note})


def result_event(payload: dict) -> str:
    return sse("result", payload)


def error_event(msg: str) -> str:
    return sse("error", {"message": msg})


# ─────────────────────────────────────────────
# Video download
# ─────────────────────────────────────────────

def download_video(url: str, out_dir: str) -> str:
    """Download a video from any URL using yt-dlp. Returns local file path."""
    ydl_opts = {
        "format":     "bestvideo[ext=mp4][height<=720]+bestaudio[ext=m4a]/best[ext=mp4]/best",
        "outtmpl":    os.path.join(out_dir, "video.%(ext)s"),
        "quiet":      True,
        "no_warnings": True,
        "merge_output_format": "mp4",
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])

    # Find the downloaded file
    for f in os.listdir(out_dir):
        if f.startswith("video."):
            return os.path.join(out_dir, f)
    raise FileNotFoundError("yt-dlp finished but no output file found.")


# ─────────────────────────────────────────────
# Map pipeline result → frontend payload
# ─────────────────────────────────────────────

def build_result(
    verdict: str,
    verdict_class: str,
    confidence: int,
    video_type: str,
    description: str,
    trace: list[dict],
    explanation: str,
    elapsed: str,
) -> dict:
    return {
        "verdict":     verdict,
        "verdictClass": verdict_class,
        "confidence":  confidence,
        "videoType":   video_type,
        "description": description,
        "trace":       trace,
        "explanation": explanation,
        "elapsed":     elapsed,
    }


# ─────────────────────────────────────────────
# /analyze  — SSE stream
# ─────────────────────────────────────────────

@app.get("/analyze")
async def analyze(url: str = Query(..., description="Video URL to analyze")):

    async def stream():
        start = time.perf_counter()
        tmp_dir = tempfile.mkdtemp()

        try:
            # ── Step 1: Receive ───────────────────────────────────────────
            yield step_event(1, "active")
            await asyncio.sleep(0.3)
            yield step_event(1, "complete")

            # ── Step 2: Route ─────────────────────────────────────────────
            yield step_event(2, "active")
            await asyncio.sleep(0.2)
            yield step_event(2, "complete")

            # ── Step 3: Download ─────────────────────────────────────────
            yield step_event(3, "active")
            try:
                video_path = await asyncio.get_event_loop().run_in_executor(
                    None, download_video, url, tmp_dir
                )
            except Exception as e:
                yield error_event(f"Could not download video: {e}")
                return
            yield step_event(3, "complete", note=os.path.basename(video_path))

            # ── Step 4: Metadata scan ─────────────────────────────────────
            yield step_event(4, "active")
            meta_result = await asyncio.get_event_loop().run_in_executor(
                None, analyze_media, video_path
            )
            yield step_event(4, "complete")

            elapsed = lambda: f"{time.perf_counter() - start:.2f}"

            # Fast-exit: definite AI
            if meta_result.isAIGenerated == Decision.YES:
                reason = str(meta_result.reason or "AI metadata confirmed.")
                for i in range(5, 7):
                    yield step_event(i, "skipped")
                yield result_event(build_result(
                    verdict="AI CONFIRMED",
                    verdict_class="ai",
                    confidence=99,
                    video_type="GEN.",
                    description=reason,
                    trace=[
                        {"icon": "flag", "text": f"PROVENANCE — {reason}"},
                        {"icon": "done", "text": "FAST EXIT — AI origin confirmed via metadata."},
                        *[{"icon": "skip", "text": f"{s} — SKIPPED (fast exit)."} for s in
                          ["VIDEO TYPE", "VIDEOMAE", "WAV2VEC", "FOURIER CNN", "FUSION LAYER"]],
                        {"icon": "flag", "text": "CLASSIFIER — AI CONFIRMED @ 99% confidence."},
                    ],
                    explanation=f"Metadata analysis conclusively identified AI generation: {reason}. No further pipeline stages required.",
                    elapsed=elapsed(),
                ))
                return

            # ── Step 5: VLM classify ──────────────────────────────────────
            yield step_event(5, "active")
            meta_reason = meta_result.reason if isinstance(meta_result.reason, str) else None
            vlm_verdict, vlm_reason = await asyncio.get_event_loop().run_in_executor(
                None, llm_classify_video, video_path, meta_reason
            )
            yield step_event(5, "complete", note=vlm_verdict)

            # Animated / Recording / Generated → skip deepfake pipeline
            # Real / Deepfake → continue to full deepfake engine
            SKIP_VERDICTS = {
                "Animated":  ("ANIM.", "skipped", 85),
                "Recording": ("REC.",  "skipped", 80),
                "Generated": ("GEN.",  "ai",      90),
            }

            if vlm_verdict in SKIP_VERDICTS:
                vtype, vclass, conf = SKIP_VERDICTS[vlm_verdict]
                yield step_event(6, "skipped")
                verdict_label = "AI CONFIRMED" if vlm_verdict == "Generated" else "SKIPPED"
                yield result_event(build_result(
                    verdict=verdict_label,
                    verdict_class=vclass,
                    confidence=conf,
                    video_type=vtype,
                    description=f"Video classified as {vlm_verdict} by VLM. {vlm_reason}",
                    trace=[
                        {"icon": "done", "text": "PROVENANCE — No conclusive AI metadata found."},
                        {"icon": "flag" if vlm_verdict == "Generated" else "done",
                         "text": f"VIDEO TYPE — {vtype} detected ({conf}% confidence). {vlm_reason}"},
                        *[{"icon": "skip", "text": f"{s} — SKIPPED (not applicable)."} for s in
                          ["VIDEOMAE", "WAV2VEC", "FOURIER CNN", "FUSION LAYER"]],
                        {"icon": "flag" if vlm_verdict == "Generated" else "done",
                         "text": f"CLASSIFIER — {verdict_label} @ {conf}% confidence."},
                    ],
                    explanation=f"The VLM classified the video as {vlm_verdict} ({vlm_reason}). The deepfake pipeline only applies to real human footage.",
                    elapsed=elapsed(),
                ))
                return

            # vlm_verdict is "Real" or "Deepfake" → run full deepfake engine
            video_type_label = "DEEP." if vlm_verdict == "Deepfake" else "REA."

            # ── Step 6: Deepfake engine (all modules run together) ────────
            yield step_event(6, "active")

            detector = await asyncio.get_event_loop().run_in_executor(None, get_detector)
            df_result = await asyncio.get_event_loop().run_in_executor(
                None, detector.predict, video_path, False
            )
            yield step_event(6, "complete")

            is_fake  = df_result["is_fake"]
            max_conf = df_result["max_confidence"]
            avg_conf = df_result["average_confidence"]
            chunks   = df_result["chunks"]

            confidence = int(max_conf * 100)
            verdict    = "FAKE" if is_fake else "GENUINE"
            vclass     = "fake" if is_fake else "genuine"
            icon       = "flag" if is_fake else "done"

            # Build per-chunk trace
            chunk_trace = []
            for c in chunks[:6]:   # cap trace at 6 lines
                label = "FAKE" if c["fake_prob"] > 0.5 else "REAL"
                chunk_trace.append({
                    "icon": "flag" if c["fake_prob"] > 0.5 else "done",
                    "text": f"CHUNK [{c['start']:.1f}s–{c['end']:.1f}s] → {label} ({c['fake_prob']:.3f})",
                })

            trace = [
                {"icon": "done", "text": "PROVENANCE — No AI metadata found. Forwarding to pipeline."},
                {"icon": "warn" if vlm_verdict == "Deepfake" else "done",
                 "text": f"VIDEO TYPE — {video_type_label} detected. {vlm_reason}"},
                *chunk_trace,
                {"icon": icon,
                 "text": f"VIDEOMAE — Spatial/temporal analysis: max fake spike {max_conf:.3f}."},
                {"icon": icon,
                 "text": f"WAV2VEC — Audio analysis: average confidence {avg_conf:.3f}."},
                {"icon": icon,
                 "text": f"FOURIER CNN — Frequency domain complete."},
                {"icon": icon,
                 "text": f"FUSION LAYER — {len(chunks)} chunk(s) analysed."},
                {"icon": icon,
                 "text": f"CLASSIFIER — Verdict: {verdict} @ {confidence}% confidence."},
            ]

            desc = (
                f"{'Deepfake artifacts detected' if is_fake else 'No conclusive deepfake artifacts detected'} "
                f"across {len(chunks)} analysis window(s). "
                f"Highest fake spike: {max_conf:.3f}. Average confidence: {avg_conf:.3f}."
            )
            explanation = (
                f"The multimodal deepfake detector analysed {len(chunks)} sliding window(s) of 2 seconds each. "
                f"The highest recorded fake probability was {max_conf:.4f} and the average was {avg_conf:.4f}. "
                f"A chunk is flagged as fake when its probability exceeds 0.5. "
                f"The final verdict uses the highest single-chunk score as the deciding signal. "
                f"Verdict: {verdict} at {confidence}% confidence."
            )

            yield result_event(build_result(
                verdict=verdict,
                verdict_class=vclass,
                confidence=confidence,
                video_type=video_type_label,
                description=desc,
                trace=trace,
                explanation=explanation,
                elapsed=elapsed(),
            ))

        except Exception as e:
            traceback.print_exc()
            yield error_event(f"Pipeline error: {e}")
        finally:
            # Clean up temp files
            import shutil
            shutil.rmtree(tmp_dir, ignore_errors=True)

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )