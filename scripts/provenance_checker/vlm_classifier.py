import cv2
import base64
import os
import time
import json
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8001/v1",
    api_key="none"
)


def get_video_frames_base64(video_path: str, num_frames: int = 60) -> list[str]:
    """
    Extract evenly-spaced frames from a video and return as base64 JPEG strings.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    indices = [int(total_frames * (i + 0.5) / num_frames) for i in range(num_frames)]

    b64_frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if ok:
            frame = cv2.resize(frame, (640, 360))
            _, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            b64_frames.append(base64.b64encode(buf).decode('utf-8'))

    cap.release()
    return b64_frames


def classify_video(video_path, reason=None):
    print(f"🎥 Processing video: {video_path}")

    # Only pass actual C2PA/XMP metadata findings as context.
    # Never pass URLs or filenames — the VLM will hallucinate from them
    # instead of analysing the actual frames.
    if reason is None or (isinstance(reason, str) and 'metadata' in reason.lower()):
        reason_str = "No AI-indicative metadata found."
    elif isinstance(reason, dict):
        reason_str = json.dumps(reason)
    else:
        reason_str = str(reason)

    try:
        frames = get_video_frames_base64(video_path, num_frames=60)
        print(f"⏳ Extracted {len(frames)} frames for analysis...")
    except Exception as e:
        print(f"❌ Error extracting frames: {e}")
        return ("Real", str(e))

    print(f"🧠 Sending {len(frames)} frames to Qwen3-VL via llama-server...")

    prompt_text = (
        "You are a forensic multimedia analyzer. "
        f"Analyze these {len(frames)} evenly-spaced frames from a video. "
        "Base your verdict ONLY on what you can visually observe in the frames — "
        "lighting, skin texture, facial movement, background, rendering style, etc. "
        "Do NOT make assumptions based on file names, URLs, video IDs, or platform names. "
        "You must respond ONLY with a valid JSON object. No markdown, no extra text. "
        "Schema:\n"
        '{"verdict": "Real" | "Animated" | "Recording" | "Deepfake" | "Generated", '
        '"reason": "1-2 sentence explanation based only on visual evidence."}\n'
        "Rules:\n"
        "- Real: live-action footage of real people or places. Use this when uncertain.\n"
        "- Animated: cartoon, CGI, anime, 3D rendered — visually obvious non-photorealistic style.\n"
        "- Recording: screen recording, gameplay, slideshow, no human face visible.\n"
        "- Deepfake: photorealistic human face with subtle visual artifacts around face/neck edges.\n"
        "- Generated: fully AI-generated photorealistic video with uncanny lighting or impossible scenes.\n"
        "Never respond with Unknown. If uncertain, pick Real."
    )

    content = []
    for b64 in frames:
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{b64}"}
        })
    content.append({
        "type": "text",
        "text": f"Metadata context: {reason_str}\n\n{prompt_text}"
    })

    try:
        response = client.chat.completions.create(
            model="qwen3vl",
            messages=[{"role": "user", "content": content}],
            max_tokens=256,
            temperature=0.2,
            extra_body={"thinking": False}
        )

        raw_result = response.choices[0].message.content.strip()

        result_dict = json.loads(raw_result)
        classification = result_dict.get('verdict', 'Real')
        valid = {'Real', 'Animated', 'Recording', 'Deepfake', 'Generated'}
        if classification not in valid:
            classification = 'Real'
        explanation = result_dict.get('reason', 'No reason provided.')

        print(f"📊 Classification Result: {classification}")
        return (classification, explanation)

    except json.JSONDecodeError:
        print(f"❌ LLM did not return valid JSON. Raw: {raw_result}")
        return ("Real", raw_result)
    except Exception as e:
        print(f"❌ Error during inference: {e}")
        return ("Real", str(e))


def llm_classify_video(file_path, reason=None):
    start_time = time.perf_counter()
    result = classify_video(file_path, reason)
    elapsed = time.perf_counter() - start_time
    print(f"Elapsed time: {elapsed:.4f} seconds")
    return result