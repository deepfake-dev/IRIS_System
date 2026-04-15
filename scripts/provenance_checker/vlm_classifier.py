# ==============================================================================
# Copyright (c) 2026 Batangas State University (The National Engineering University)
# Project: IRIS Assistant System - Multimedia Provenance
# 
# Attributions:
# - Vision-Language Reasoning powered by Gemma / Qwen (via OpenAI API bindings)
# ==============================================================================

import cv2
import math
import base64
import socket
import json
from openai import OpenAI

class VLMClassifier:
    def __init__(self, client, vlm_alias):
        self.vlm_alias = vlm_alias

        self.client = client

    def get_chunked_video_frames(self, video_path: str, chunk_sec: float = 2.0, frames_per_chunk: int = 4) -> list[dict]:
        """Slices the video into temporal chunks and extracts frames for VLM analysis."""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if fps == 0 or total_frames == 0:
            cap.release()
            return []

        duration = total_frames / fps
        num_chunks = math.ceil(duration / chunk_sec)
        chunked_data = []

        for i in range(num_chunks):
            start_time = i * chunk_sec
            end_time = min((i + 1) * chunk_sec, duration)
            start_f = int(start_time * fps)
            end_f = int(end_time * fps)
            chunk_len = end_f - start_f

            if chunk_len <= 0: 
                continue

            indices = [start_f + int(chunk_len * (j + 0.5) / frames_per_chunk) for j in range(frames_per_chunk)]
            
            current_chunk_frames = []
            for idx in indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, min(idx, total_frames - 1))
                ok, frame = cap.read()
                if ok:
                    frame = cv2.resize(frame, (640, 360))
                    _, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                    current_chunk_frames.append(base64.b64encode(buf).decode('utf-8'))
            
            if current_chunk_frames:
                chunked_data.append({
                    "chunk_index": i + 1,
                    "start": start_time,
                    "end": end_time,
                    "frames": current_chunk_frames
                })

        cap.release()
        return chunked_data

    def classify_chunk(self, frames_b64: list, chunk_index: int, start_time: float, end_time: float, reason=None):
        """Evaluates a single 2-second window of the video."""
        reason_str = json.dumps(reason) if isinstance(reason, dict) else str(reason or "No AI metadata found.")

        prompt_text = (
            f"You are a forensic multimedia analyzer evaluating a 2-second window of a video (from {start_time:.1f}s to {end_time:.1f}s). "
            "You are provided with 4 sequential frames from this exact window. "
            "Look at these images carefully and classify the window into exactly one of these three categories:\n\n"
            "- ANIMATED: cartoons, anime, CGI, illustrated, drawn, painted, video game footage, game engines, virtual characters, 3D rendered scenes, game HUDs or UI overlays, stylized or non-photorealistic visuals of any kind\n"
            "- REAL_NO_HUMANS: real-world photography or video footage with no people present, must look like it was captured by a real camera\n"
            "- REAL_WITH_HUMANS: real-world photography or video footage of actual real human beings, must look like it was captured by a real camera\n\n"
            "IMPORTANT: If the images contain any game UI, anime-style art, or non-photorealistic characters — even if they look human — classify as ANIMATED.\n\n"
            "You MUST respond ONLY with a valid JSON object. No markdown, no extra text.\n"
            'Schema: {"verdict": "ANIMATED" | "REAL_NO_HUMANS" | "REAL_WITH_HUMANS", "reason": "Short explanation."}'
        )

        content = []
        for b64 in frames_b64:
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{b64}"}
            })
                
        content.append({
            "type": "text",
            "text": f"Metadata context: {reason_str}\n\n{prompt_text}"
        })

        try:
            response = self.client.chat.completions.create(
                model=self.vlm_alias,
                messages=[{"role": "user", "content": content}],
            )
            result_dict = json.loads(response.choices[0].message.content.strip())
            
            classification = result_dict.get('verdict', 'REAL_WITH_HUMANS')
            if classification not in {'ANIMATED', 'REAL_NO_HUMANS', 'REAL_WITH_HUMANS'}:
                classification = 'REAL_WITH_HUMANS'
                
            return (classification, result_dict.get('reason', 'No reason provided.'))

        except Exception as e:
            return ("REAL_NO_HUMANS", f"Error evaluating window: {str(e)}")