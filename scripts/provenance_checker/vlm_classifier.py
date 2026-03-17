import cv2
import base64
import os
import time
import json
from openai import OpenAI

# Point to your llama-server VL instance
client = OpenAI(
    base_url="http://localhost:8001/v1",  # separate port for VL model
    api_key="none"
)

def get_video_frames_base64(video_path, num_frames=3):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_indices = [int(total_frames * i / num_frames) for i in range(num_frames)]

    base64_frames = []
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        success, frame = cap.read()
        if success:
            frame_resized = cv2.resize(frame, (640, 360))
            _, buffer = cv2.imencode('.jpg', frame_resized)
            b64_string = base64.b64encode(buffer).decode('utf-8')
            base64_frames.append(b64_string)

    cap.release()
    return base64_frames

def classify_video(video_path, reason=None):
    print(f"🎥 Processing video: {video_path}")
    print("⏳ Extracting frames...")

    if reason is None:
        reason_str = f"File name: {os.path.basename(video_path)}"
    elif isinstance(reason, dict):
        reason['file_name'] = os.path.basename(video_path)
        reason_str = json.dumps(reason)
    else:
        reason_str = str(reason)

    try:
        frames = get_video_frames_base64(video_path, num_frames=3)
    except Exception as e:
        print(f"❌ Error extracting frames: {e}")
        return ("Unknown", str(e))

    print("🧠 Sending to Qwen3-VL via llama-server...")

    prompt_text = (
        "You are a forensic multimedia analyzer. "
        "Analyze these 3 sequential frames from a video. Determine if the visual content is "
        "a real-world live-action recording, AI-generated, a Deepfake, a screen-recording, or animated/CGI. "
        "Pay special attention to the provided metadata context. "
        "You must respond ONLY with a valid JSON object. No markdown, no extra text. "
        "Schema:\n"
        '{"verdict": "Real" | "Animated" | "Recording" | "Deepfake" | "Generated", '
        '"reason": "1-2 sentence explanation."}'
    )

    # Build content with interleaved images (qwen-vl-utils style)
    image_content = []
    for b64 in frames:
        image_content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{b64}"}
        })

    image_content.append({
        "type": "text",
        "text": f"Metadata/Context: {reason_str}\n\n{prompt_text}"
    })

    try:
        response = client.chat.completions.create(
            model="qwen3vl",
            messages=[{"role": "user", "content": image_content}],
            max_tokens=256,
            temperature=0.2,
            extra_body={"thinking": False}
        )

        raw_result = response.choices[0].message.content.strip()

        result_dict = json.loads(raw_result)
        classification = result_dict.get('verdict', 'Unknown')
        explanation = result_dict.get('reason', 'No reason provided.')

        print(f"📊 Classification Result: {classification}")
        return (classification, explanation)

    except json.JSONDecodeError:
        print(f"❌ LLM did not return valid JSON. Raw: {raw_result}")
        return ("Unknown", raw_result)
    except Exception as e:
        print(f"❌ Error during inference: {e}")
        return ("Unknown", str(e))

def llm_classify_video(file_path, reason=None):
    start_time = time.perf_counter()
    result = classify_video(file_path, reason)
    elapsed = time.perf_counter() - start_time
    print(f"Elapsed time: {elapsed:.4f} seconds")
    return result