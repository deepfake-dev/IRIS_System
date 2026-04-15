# ==============================================================================
# Copyright (c) 2026 Batangas State University (The National Engineering University)
# Project: IRIS Assistant System
# 
# Attributions:
# - Wake Word Detection powered by OpenWakeWord (David Scripka)
# - Audio generation utilizes Kokoro TTS (StyleTTS2 architecture)
# ==============================================================================

import io
import wave
import json
import queue
import asyncio
import threading
import numpy as np
import logging

from openwakeword.model import Model

class ClientSession:
    """
    Manages state, audio buffers, and concurrent TTS generation for a SINGLE kiosk.
    Isolates client-specific memory to prevent multi-tenant data leakage.
    """
    def __init__(self, websocket, loop, iris_engine):
        self.ws = websocket
        self.loop = loop
        self.iris = iris_engine
        
        self.interrupt = False
        self.tts_voice = self.iris.female_voice
        self.audio_buffer = np.array([], dtype=np.int16)
        self.kiosk_location = "Unknown Location"

        # Initialize Wake Word Model
        try:
            self.oww_model = Model(wakeword_models=[self.iris.wake_word_path], inference_framework="onnx")
        except ValueError as e:
            logging.warning(f"Failed to load specific wake word, falling back to default: {e}")
            self.oww_model = Model(wakeword_models=["hey_jarvis"], inference_framework="onnx")
        
        # Concurrent TTS Queue for THIS specific client
        self.tts_queue = queue.Queue()
        self.worker = threading.Thread(target=self._tts_worker, daemon=True)
        self.worker.start()

    def _tts_worker(self):
        """Background thread that constantly generates audio without blocking the LLM."""
        while True:
            item = self.tts_queue.get()
            if item is None: 
                break  # Shutdown signal received
            
            text, display_text = item
            try:
                # Generate audio arrays via Kokoro
                samples, sample_rate = self.iris.tts.create(text, voice=self.tts_voice, speed=1.0, lang="en-us")
                samples = np.clip(samples, -1.0, 1.0)
                
                # Convert raw arrays to standard WAV bytes
                buf = io.BytesIO()
                with wave.open(buf, 'wb') as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2)
                    wf.setframerate(sample_rate)
                    wf.writeframes((samples * 32767).astype(np.int16).tobytes())
                wav_bytes = buf.getvalue()
                
                # Transmit concurrently to the frontend
                asyncio.run_coroutine_threadsafe(self.ws.send(json.dumps({"ai_text_sync": display_text})), self.loop)
                asyncio.run_coroutine_threadsafe(self.ws.send(wav_bytes), self.loop)
                
            except Exception as e:
                logging.error(f"TTS Engine Error: {e}")
            finally:
                self.tts_queue.task_done()

    def on_sentence_ready(self, spoken_text, display_text):
        """Callback fired by the LLM Engine to push text to the audio generator."""
        self.tts_queue.put((spoken_text, display_text))

    def check_interrupt(self):
        """Checks if the user has requested to skip the current response."""
        return self.interrupt

    def clear_queue(self):
        """Instantly drops pending text if the user hits Skip/Interrupt."""
        while not self.tts_queue.empty():
            try:
                self.tts_queue.get_nowait()
                self.tts_queue.task_done()
            except queue.Empty:
                break