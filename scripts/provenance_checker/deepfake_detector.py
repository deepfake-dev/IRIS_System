# ==============================================================================
# Copyright (c) 2026 Batangas State University (The National Engineering University)
# Project: IRIS Assistant System - Multimedia Provenance
# ==============================================================================

import os
import math
import numpy as np
import cv2
import tempfile
import onnxruntime as ort
import soundfile as sf
from decord import VideoReader, cpu
from moviepy import VideoFileClip

class DeepfakeDetector:
    def __init__(self, onnx_path: str, providers: list = None):
        print("🚀 Initializing ONNX Deepfake Detector...")
        if providers is None:
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        self.ort_session = ort.InferenceSession(onnx_path, providers=providers)
        
        self.num_frames = 16
        self.audio_duration = 2.0
        self.sample_rate = 16000
        self.target_audio_len = int(self.sample_rate * self.audio_duration)
        self.frame_size = 224
        self.img_mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.img_std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def _preprocess_video(self, frames_list: list) -> np.ndarray:
        processed = []
        for frame in frames_list:
            resized = cv2.resize(frame, (self.frame_size, self.frame_size))
            normalized = (resized.astype(np.float32) / 255.0 - self.img_mean) / self.img_std
            transposed = np.transpose(normalized, (2, 0, 1))
            processed.append(transposed)
        return np.expand_dims(np.stack(processed), axis=0).astype(np.float32)

    def _preprocess_audio_chunk(self, audio_chunk: np.ndarray) -> tuple:
        original_len = len(audio_chunk)
        if original_len < self.target_audio_len:
            audio_chunk = np.pad(audio_chunk, (0, self.target_audio_len - original_len), mode='constant')
            
        mean = np.mean(audio_chunk)
        var = np.var(audio_chunk)
        normalized = (audio_chunk - mean) / np.sqrt(var + 1e-7)
        
        if np.std(normalized) < 1e-6:
            normalized = np.random.randn(self.target_audio_len).astype(np.float32) * 1e-5
            
        a_mask = np.zeros(self.target_audio_len, dtype=np.int64)
        a_mask[:min(original_len, self.target_audio_len)] = 1
        return normalized.astype(np.float32), a_mask

    @staticmethod
    def get_fourier_map(frame: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        gray_small = cv2.resize(gray, (224, 224)).astype(np.float32)
        f = np.fft.fft2(gray_small)
        fshift = np.fft.fftshift(f)
        magnitude = 20 * np.log(np.abs(fshift) + 1)
        mag_norm = (magnitude - magnitude.min()) / (magnitude.max() - magnitude.min() + 1e-8)
        return mag_norm[np.newaxis, np.newaxis, ...].astype(np.float32)

    def load_media(self, video_path: str):
        """Loads the video/audio into memory ONCE so chunks can be processed rapidly."""
        try:
            temp_wav = os.path.join(tempfile.gettempdir(), f"temp_proc_{os.getpid()}.wav")
            with VideoFileClip(video_path) as clip:
                if clip.audio is None:
                    audio_data = np.zeros(self.target_audio_len, dtype=np.float32)
                else:
                    clip.audio.write_audiofile(temp_wav, logger=None, fps=self.sample_rate)
                    audio_data, _ = sf.read(temp_wav)
                    if audio_data.ndim > 1: audio_data = np.mean(audio_data, axis=1)
                    if os.path.exists(temp_wav): os.remove(temp_wav)
        except Exception:
            audio_data = np.zeros(self.target_audio_len, dtype=np.float32)

        vr = VideoReader(video_path, ctx=cpu(0))
        return audio_data.astype(np.float32), vr, vr.get_avg_fps(), len(vr)

    def predict_window(self, audio, vr, fps, total_frames, start_time: float, end_time: float) -> float:
        """Processes a single isolated temporal window through the multimodal architecture."""
        start_sample = int(start_time * self.sample_rate)
        end_sample = start_sample + self.target_audio_len
        
        if start_sample < len(audio):
            audio_chunk = audio[start_sample:end_sample]
        else:
            audio_chunk = np.zeros(self.target_audio_len, dtype=np.float32)
            
        a_values, a_mask = self._preprocess_audio_chunk(audio_chunk)

        start_f = int(start_time * fps)
        end_f = int(end_time * fps)
        indices = np.linspace(start_f, max(start_f, end_f - 1), self.num_frames, dtype=int)
        indices = [min(idx, total_frames - 1) for idx in indices]
        
        try:
            raw_frames = vr.get_batch(indices).asnumpy()
            pixel_values = self._preprocess_video(list(raw_frames))
            mid_frame = raw_frames[len(raw_frames) // 2]
            freq_map = self.get_fourier_map(mid_frame)
        except Exception as e:
            print(f"⚠️ Skip window at {start_time}s: {e}")
            return 0.0

        ort_inputs = {
            'pixel_values': pixel_values,
            'audio_values': a_values[np.newaxis, ...],
            'audio_mask': a_mask[np.newaxis, ...],
            'freq_maps': freq_map
        }
        
        logits = self.ort_session.run(None, ort_inputs)[0]
        exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
        return float(probs[0, 1])