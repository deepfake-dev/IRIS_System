import os
import sys
import math
import time
import torch
import torch.nn as nn
import torchaudio
import numpy as np
import cv2
from decord import VideoReader, cpu
from transformers import VideoMAEModel, Wav2Vec2Model, VideoMAEImageProcessor, Wav2Vec2FeatureExtractor, logging
from moviepy import VideoFileClip
import tempfile

logging.set_verbosity(logging.CRITICAL)

# ==============================================================================
# 1. THE EXACT ARCHITECTURE
# ==============================================================================
class TriStreamDeepfakeDetector(nn.Module):
    def __init__(self):
        super().__init__()
        # Visual Branch: VideoMAE
        self.video_backbone = VideoMAEModel.from_pretrained("MCG-NJU/videomae-base")
        
        # Audio Branch: Wav2Vec2
        self.audio_backbone = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")
        
        # Frequency Branch: Fourier CNN
        self.freq_branch = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )
        
        # Cross-Modal Fusion: Sync Check
        self.sync_checker = nn.MultiheadAttention(embed_dim=768, num_heads=8, batch_first=True)
        
        # Final Classifier
        self.head = nn.Sequential(
            nn.Linear(768 + 768 + 32, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 2)
        )

    def forward(self, pixel_values, audio_values, audio_mask, freq_maps):
        v_hidden = self.video_backbone(pixel_values).last_hidden_state
        
        a_hidden = self.audio_backbone(
            audio_values,
            attention_mask=audio_mask
        ).last_hidden_state
        
        f_feats = self.freq_branch(freq_maps)
        
        sync_context, _ = self.sync_checker(a_hidden, v_hidden, v_hidden)
        
        v_pool = torch.mean(v_hidden, dim=1)
        sync_pool = torch.mean(sync_context, dim=1)
        
        combined = torch.cat([v_pool, sync_pool, f_feats], dim=1)
        return self.head(combined)

# ==============================================================================
# 2. PREPROCESSING HELPERS
# ==============================================================================
def get_fourier_map(frame: np.ndarray) -> torch.Tensor:
    """Generates the FFT magnitude spectrum from the middle frame."""
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    gray_small = cv2.resize(gray, (224, 224)) 
    
    f = np.fft.fft2(gray_small)
    fshift = np.fft.fftshift(f)
    magnitude = 20 * np.log(np.abs(fshift) + 1)
    
    return torch.tensor(magnitude, dtype=torch.float32).unsqueeze(0) / 255.0

# ==============================================================================
# 3. SLIDING WINDOW INFERENCE
# ==============================================================================
def process_full_video(video_path: str, checkpoint_path: str):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 Initializing PyTorch using device: {device}")
    
    # 1. Load Model
    model = TriStreamDeepfakeDetector()
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint.get('model_state_dict', checkpoint))
    model.to(device)
    model.eval()
    print("✅ Model weights loaded successfully!")

    # 2. Load Processors
    print("⏳ Loading HuggingFace Processors...")
    v_processor = VideoMAEImageProcessor.from_pretrained("MCG-NJU/videomae-base")
    a_processor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/wav2vec2-base-960h")

    # Configuration mapped from your training parameters
    num_frames = 16
    audio_duration = 2.0
    audio_sample_rate = 16000
    target_audio_len = int(audio_sample_rate * audio_duration)

    print(f"\n📂 Analyzing Video: {video_path}")
    
    # --- Load Audio ---
    # --- Load Audio (Bulletproof Workaround) ---
    try:
        # 1. Safely extract audio to a temporary WAV file
        temp_wav = os.path.join(tempfile.gettempdir(), "temp_deepfake_audio.wav")
        with VideoFileClip(video_path) as video_clip:
            if video_clip.audio is None:
                raise ValueError("Video contains no audio track")
            # Write to WAV silently
            video_clip.audio.write_audiofile(temp_wav, logger=None, fps=audio_sample_rate)
            
        # 2. Load the clean WAV with torchaudio
        audio, orig_sr = torchaudio.load(temp_wav)
        
        # 3. Process and Cleanup
        if audio.shape[0] > 1:
            audio = audio.mean(dim=0, keepdim=True)
        if orig_sr != audio_sample_rate:
            audio = torchaudio.functional.resample(audio, orig_sr, audio_sample_rate)
        total_audio_duration = audio.shape[1] / audio_sample_rate
        
        os.remove(temp_wav) # Clean up the temp file
        
    except Exception as e:
        print(f"⚠️ Could not extract audio. Padding with zeros. Error: {e}")
        audio = torch.zeros(1, target_audio_len)
        total_audio_duration = 0

    # --- Load Video ---
    vr = VideoReader(video_path, ctx=cpu(0))
    fps = vr.get_avg_fps()
    total_frames = len(vr)
    total_video_duration = total_frames / fps
    print(f"⏱️ Video Duration: {total_video_duration:.2f}s | FPS: {fps:.2f} | Total Frames: {total_frames}")

    duration = max(total_video_duration, total_audio_duration)
    num_chunks = math.ceil(duration / audio_duration)
    chunk_scores = []

    print("\n" + "="*50 + "\n🔍 COMMENCING SLIDING WINDOW ANALYSIS\n" + "="*50)

    with torch.no_grad():
        for i in range(num_chunks):
            start_time = i * audio_duration
            end_time = min((i + 1) * audio_duration, duration)
            
            # --- EXTRACT AUDIO CHUNK ---
            start_sample = int(start_time * audio_sample_rate)
            end_sample = start_sample + target_audio_len
            
            if start_sample >= audio.shape[1]:
                audio_chunk = torch.zeros(target_audio_len)
            else:
                audio_chunk = audio[0, start_sample:end_sample]
                
            if len(audio_chunk) < target_audio_len:
                padding = target_audio_len - len(audio_chunk)
                audio_chunk = torch.nn.functional.pad(audio_chunk, (0, padding))

            # Handle absolute silence to prevent NaN
            if audio_chunk.numel() == 0 or torch.all(audio_chunk == 0) or audio_chunk.std() < 1e-6:
                audio_chunk = torch.randn(target_audio_len) * 1e-5

            audio_features = a_processor(
                audio_chunk.numpy(), 
                sampling_rate=audio_sample_rate, 
                return_tensors="pt",
                return_attention_mask=True
            )
            a_values = audio_features['input_values'].to(device)
            a_mask = audio_features['attention_mask'].to(device)

            # --- EXTRACT VIDEO CHUNK ---
            # --- EXTRACT VIDEO CHUNK ---
            start_frame = int(start_time * fps)
            end_frame = int(end_time * fps)
            
            frame_indices = np.linspace(start_frame, end_frame - 1, num_frames, dtype=int).tolist()
            frame_indices = [min(idx, total_frames - 1) for idx in frame_indices]
            
            # --- THE FIX: Safe extraction to prevent decord from crashing ---
            try:
                raw_frames = vr.get_batch(frame_indices).asnumpy()
            except Exception as e:
                print(f"⚠️ Corrupted video frames detected between {start_time:05.2f}s - {end_time:05.2f}s. Skipping chunk.")
                continue # Skip to the next 2-second chunk
            
            pixel_values = v_processor(list(raw_frames), return_tensors="pt")['pixel_values'].to(device)
            
            # --- GENERATE FREQ MAP ---
            middle_frame = raw_frames[len(raw_frames)//2]
            freq_map = get_fourier_map(middle_frame).unsqueeze(0).to(device)

            # --- INFERENCE ---
            logits = model(pixel_values, a_values, a_mask, freq_map)
            
            # Calculate Softmax for CrossEntropy format (Fake is index 1)
            probabilities = torch.softmax(logits, dim=1)
            fake_prob = probabilities[0, 1].item()
            
            chunk_scores.append(fake_prob)
            
            status = "🛑 FAKE" if fake_prob > 0.5 else "✅ REAL"
            print(f"⏱️ [{start_time:05.2f}s - {end_time:05.2f}s] -> {status:<6} (Conf: {fake_prob:.4f})")

    # --- AGGREGATION & FINAL VERDICT ---
    print("\n" + "="*50 + "\n📊 FINAL ANALYSIS\n" + "="*50)
    
    avg_conf = sum(chunk_scores) / len(chunk_scores)
    max_conf = max(chunk_scores)
    
    is_fake = max_conf > 0.5
    
    # print(f"Overall Verdict:        {'🛑 FAKE' if is_fake else '✅ REAL'}")
    # print(f"Highest Fake Spike:     {max_conf:.4f} (Used for final verdict)")
    # print(f"Average AI Confidence:  {avg_conf:.4f} (For reference only)")
    return is_fake, max_conf, avg_conf

def process_video_file(file_path):
    VIDEO_FILE = file_path
    CHECKPOINT_FILE = "../current_best_models/best_model_6.pt"

    start_time = time.perf_counter()
    verdict = process_full_video(VIDEO_FILE, CHECKPOINT_FILE)
    end_time = time.perf_counter()

    elapsed_time = end_time - start_time
    print(f"\nTime taken: {elapsed_time:.2f} seconds")

    torch.cuda.empty_cache()

    return verdict

# if __name__ == "__main__":
#     if len(sys.argv) < 2:
#         print("Usage: python test_vit_deepfake.py <path_to_video> <path_to_checkpoint>")
#         sys.exit(1)
        
#     VIDEO_FILE = sys.argv[1]
#     CHECKPOINT_FILE = "current_best_models/best_model_6.pt"
    
#     start_time = time.perf_counter()
#     process_full_video(VIDEO_FILE, CHECKPOINT_FILE)
#     end_time = time.perf_counter()

#     elapsed_time = end_time - start_time
#     print(f"\nTime taken: {elapsed_time:.2f} seconds")