import os
import math
import torch
import torchaudio
import numpy as np
import cv2
import tempfile
import onnxruntime as ort
from decord import VideoReader, cpu
from transformers import VideoMAEImageProcessor, Wav2Vec2FeatureExtractor
from moviepy import VideoFileClip

class DeepfakeDetector:
    def __init__(self, onnx_path: str, device: str = None):
        """
        Initializes the detector, loading models and processors into memory once.
        """
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        print(f"🚀 Initializing Detector on: {self.device}")

        # 1. Load ONNX Session
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if self.device.type == 'cuda' else ['CPUExecutionProvider']
        self.ort_session = ort.InferenceSession(onnx_path, providers=providers)
        
        # 2. Load Processors from Hugging Face
        print("⏳ Loading HuggingFace Processors (VideoMAE & Wav2Vec2)...")
        self.v_processor = VideoMAEImageProcessor.from_pretrained("MCG-NJU/videomae-base")
        self.a_processor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/wav2vec2-base-960h")

        # Config Constants derived from training parameters
        self.num_frames = 16
        self.audio_duration = 2.0
        self.audio_sample_rate = 16000
        self.target_audio_len = int(self.audio_sample_rate * self.audio_duration)

    @staticmethod
    def get_fourier_map(frame: np.ndarray) -> np.ndarray:
        """Generates the FFT magnitude spectrum for frequency analysis."""
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        gray_small = cv2.resize(gray, (224, 224)) 
        f = np.fft.fft2(gray_small)
        fshift = np.fft.fftshift(f)
        magnitude = 20 * np.log(np.abs(fshift) + 1)
        # Returns (1, 1, 224, 224) shape for ONNX input
        return (magnitude / 255.0)[np.newaxis, np.newaxis, ...].astype(np.float32)

    def _extract_audio(self, video_path: str):
        """Extracts audio to a temporary WAV file for consistent processing."""
        try:
            temp_wav = os.path.join(tempfile.gettempdir(), f"temp_proc_{os.getpid()}.wav")
            with VideoFileClip(video_path) as video_clip:
                if video_clip.audio is None:
                    return torch.zeros(1, self.target_audio_len), 0
                video_clip.audio.write_audiofile(temp_wav, logger=None, fps=self.audio_sample_rate)
            
            audio, orig_sr = torchaudio.load(temp_wav)
            if audio.shape[0] > 1:
                audio = audio.mean(dim=0, keepdim=True)
            
            if os.path.exists(temp_wav):
                os.remove(temp_wav)
            return audio, audio.shape[1] / self.audio_sample_rate
        except Exception as e:
            print(f"⚠️ Audio extraction error: {e}")
            return torch.zeros(1, self.target_audio_len), 0

    def predict(self, video_path: str, verbose: bool = True):
        """
        Runs sliding window inference on the video.
        """
        audio, total_audio_duration = self._extract_audio(video_path)
        vr = VideoReader(video_path, ctx=cpu(0))
        fps = vr.get_avg_fps()
        total_frames = len(vr)
        total_video_duration = total_frames / fps

        duration = max(total_video_duration, total_audio_duration)
        num_chunks = math.ceil(duration / self.audio_duration)
        chunk_results = []

        print(f"🔍 Analyzing {num_chunks} chunks of 2.0s each...")

        for i in range(num_chunks):
            start_time = i * self.audio_duration
            end_time = min((i + 1) * self.audio_duration, duration)

            # --- AUDIO PROCESSING ---
            start_sample = int(start_time * self.audio_sample_rate)
            end_sample = start_sample + self.target_audio_len
            
            if start_sample < audio.shape[1]:
                audio_chunk = audio[0, start_sample:end_sample]
            else:
                audio_chunk = torch.zeros(self.target_audio_len)
            
            if len(audio_chunk) < self.target_audio_len:
                audio_chunk = torch.nn.functional.pad(audio_chunk, (0, self.target_audio_len - len(audio_chunk)))
            
            # Prevent NaN: ensure some variance in the audio signal
            if audio_chunk.std() < 1e-6:
                audio_chunk = torch.randn(self.target_audio_len) * 1e-5

            # Fix: Explicitly request attention_mask and check for it
            a_feats = self.a_processor(
                audio_chunk.numpy(), 
                sampling_rate=self.audio_sample_rate, 
                return_tensors="pt",
                return_attention_mask=True
            )
            
            a_values = a_feats['input_values'].numpy()
            a_mask = a_feats.get('attention_mask', torch.ones_like(a_feats['input_values'])).numpy().astype(np.int64)

            # --- VIDEO PROCESSING ---
            start_f = int(start_time * fps)
            end_f = int(end_time * fps)
            indices = np.linspace(start_f, max(start_f, end_f - 1), self.num_frames, dtype=int)
            indices = [min(idx, total_frames - 1) for idx in indices]
            
            try:
                raw_frames = vr.get_batch(indices).asnumpy()
                pixel_values = self.v_processor(list(raw_frames), return_tensors="pt")['pixel_values'].numpy()
                freq_map = self.get_fourier_map(raw_frames[len(raw_frames)//2])
            except Exception as e:
                print(f"⚠️ Skip chunk at {start_time}s: {e}")
                continue

            # --- ONNX INFERENCE ---
            ort_inputs = {
                'pixel_values': pixel_values,
                'audio_values': a_values,
                'audio_mask': a_mask,
                'freq_map': freq_map
            }
            
            logits = self.ort_session.run(None, ort_inputs)[0]
            
            # Manual Softmax for verdict
            exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
            probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
            fake_prob = float(probs[0, 1])
            
            chunk_results.append({"start": start_time, "end": end_time, "fake_prob": fake_prob})
            
            if verbose:
                status = "🛑 FAKE" if fake_prob > 0.5 else "✅ REAL"
                print(f"⏱️ [{start_time:04.1f}s - {end_time:04.1f}s] -> {status} (Conf: {fake_prob:.4f})")

        # Final Aggregation
        max_conf = max([c['fake_prob'] for c in chunk_results]) if chunk_results else 0
        avg_conf = sum([c['fake_prob'] for c in chunk_results]) / len(chunk_results) if chunk_results else 0

        return {
            "is_fake": max_conf > 0.5,
            "max_confidence": max_conf,
            "average_confidence": avg_conf,
            "chunks": chunk_results
        }