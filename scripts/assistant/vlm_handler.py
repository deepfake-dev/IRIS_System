import asyncio
import json
import numpy as np
import warnings
import logging
import os
import faiss
import sqlite3
from openai import OpenAI

logging.getLogger().setLevel(logging.ERROR)
warnings.filterwarnings("ignore")
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

from openwakeword.model import Model
from faster_whisper import WhisperModel
from sentence_transformers import SentenceTransformer
from kokoro_onnx import Kokoro

class IrisAssistant:
    def __init__(self,
                 wake_word="hey_iris",
                 whisper_size="small",
                 tts_voice='bf_isabella',
                 # CRITICAL: Put the absolute or correct relative path to your custom model here!
                 wake_word_path='hey_iris.onnx', 
                 faiss_path="llama_index/Citizens_Charter_Handbook_2025.faiss",
                 db_path="llama_index/Citizens_Charter_Handbook_2025.db",
                 embed_model="BAAI/bge-small-en-v1.5",
                 rag_top_k=5,
                 websocket=None, 
                 websocket_loop=None):
        
        self.tts_voice = tts_voice
        self.rag_top_k = rag_top_k
        self.client = OpenAI(base_url="http://localhost:8001/v1", api_key="sk-no-key-required")

        print(f"Loading Iris Backend (Whisper {whisper_size})...")

        # --- Wake Word ---
        try:
            self.oww_model = Model(wakeword_models=[wake_word_path], inference_framework="onnx")
            self.wakeword_key = os.path.splitext(os.path.basename(wake_word_path))[0]
            print(f"✅ Loaded custom wake word: {self.wakeword_key}")
        except Exception as e:
            print(f"⚠️ Could not load {wake_word_path}: {e}")
            print("   Falling back to hey_jarvis...")
            self.oww_model = Model(wakeword_models=["hey_jarvis"], inference_framework="onnx")
            self.wakeword_key = "hey_jarvis"

        # --- Whisper ---
        self.whisper = WhisperModel(whisper_size, device="cuda", compute_type="int8")

        self.websocket = websocket
        self.websocket_loop = websocket_loop

        # --- TTS & RAG ---
        self.tts = Kokoro("kokoro-v1.0.onnx", "voices-v1.0.bin")
        self.embed_model = SentenceTransformer(embed_model)
        self.faiss_index = faiss.read_index(faiss_path)
        self.db_conn = sqlite3.connect(db_path, check_same_thread=False)

        self.speaking = False
        self.is_thinking = False # <--- ADD THIS LINE
        self._wakeword_triggered = False

        self.audio_buffer = np.array([], dtype=np.int16)

    def process_audio_chunk(self, pcm_int16: np.ndarray):
        # 1. If we are talking/thinking, flush the buffer so we don't process stale audio later
        if self.speaking or getattr(self, 'is_thinking', False):
            self.audio_buffer = np.array([], dtype=np.int16)
            return
        
        # 2. Add the new browser audio to our continuous buffer
        self.audio_buffer = np.concatenate((self.audio_buffer, pcm_int16))
        
        step = 1280
        
        # 3. Process exact 1280-sample chunks without dropping the leftovers!
        while len(self.audio_buffer) >= step:
            # Slice exactly 1280 samples
            sub_chunk = self.audio_buffer[:step]
            # Keep the remainder in the buffer for the next time the browser sends data
            self.audio_buffer = self.audio_buffer[step:]
            
            prediction = self.oww_model.predict(sub_chunk)
            score = prediction.get(self.wakeword_key, 0.0)

            if score > 0.4:
                print(f"⚡ Wake word detected! (score={score:.2f})")
                if self.websocket and self.websocket_loop:
                    asyncio.run_coroutine_threadsafe(
                        self.websocket.send(json.dumps({"wakeword": True})),
                        self.websocket_loop
                    )
                self._wakeword_triggered = True
                
                # Clear the buffer so we don't double-trigger
                self.audio_buffer = np.array([], dtype=np.int16) 
                
                # DELETED: self.oww_model.reset() -> Do NOT use this in a streaming architecture
                break

    def listen_for_command(self, pcm_chunks):
        if self.websocket and self.websocket_loop:
            asyncio.run_coroutine_threadsafe(
                self.websocket.send(json.dumps({"listening": True})),
                self.websocket_loop
            )

        # Flatten the list of arrays into a single float32 array for Whisper
        audio_flat = np.concatenate(pcm_chunks).flatten().astype(np.float32) / 32768.0
        
        print("🧠 Whisper transcribing...")
        segments, _ = self.whisper.transcribe(audio_flat, beam_size=5, language="en")
        text = "".join(seg.text for seg in segments).strip()
        print(f"📝 Heard: {text}")

        if self.websocket and self.websocket_loop:
            asyncio.run_coroutine_threadsafe(
                self.websocket.send(json.dumps({"listening": False})),
                self.websocket_loop
            )

        return text

    def _retrieve(self, query):
        q_vec = self.embed_model.encode([query], normalize_embeddings=True).astype("float32")
        _, ids = self.faiss_index.search(q_vec, self.rag_top_k)

        cur = self.db_conn.cursor()
        placeholders = ",".join("?" * len(ids[0]))
        cur.execute(f"SELECT text FROM chunks WHERE row_id IN ({placeholders})", ids[0].tolist())
        rows = cur.fetchall()
        return "\n\n".join(row[0] for row in rows)

    def speak(self, text):
        if not text.strip():
            return
            
        self.speaking = True
        
        try:
            samples, sample_rate = self.tts.create(text, voice=self.tts_voice, speed=1.0, lang="en-us")

            # 🛠️ FIX: Clamp the audio to prevent integer overflow (crackling)
            samples = np.clip(samples, -1.0, 1.0) 

            import io, wave
            buf = io.BytesIO()
            with wave.open(buf, 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(sample_rate)
                wf.writeframes((samples * 32767).astype(np.int16).tobytes())
            wav_bytes = buf.getvalue()

            if self.websocket and self.websocket_loop:
                asyncio.run_coroutine_threadsafe(
                    self.websocket.send(wav_bytes),
                    self.websocket_loop
                ).result()
                
        except Exception as e:
            print(f"⚠️ TTS Error: {e}")
            
        finally:
            self.speaking = False

    def chat(self, user_text):
        print(f"📝 You: {user_text}")

        context = self._retrieve(user_text)
        print(f"📚 Retrieved {self.rag_top_k} chunks from Citizens Charter.")

        iris_context = (
            "You are Iris, an AI assistant for Batangas State University - "
            "The National Engineering University - Alangilan Campus. "
            "Answer in pure text form, like it will be read as is, and do not format to Markdown."
            "Your birthday is February 30, 1777. You never use emojis or emoticons. "
            "You answer only specific information about yourself and Batangas State University. "
            "The President of Batangas State University (BSU) is Dr. Tirso A. Ronquillo. "
        )

        combined_payload = (
            f"{iris_context}\n\n"
            f"Use the following excerpts from the Citizens Charter Handbook to help answer "
            f"the question.\n\n"
            f"--- Citizens Charter Excerpts ---\n{context}\n"
            f"--- End of Excerpts ---\n\n"
            f"User Question: {user_text}"
        )

        print("🤖 Iris: ", end="", flush=True)
        full_response = ""
        sentence_buffer = ""

        response_stream = self.client.chat.completions.create(
            model="qwen3-vl", 
            messages=[{"role": "user", "content": combined_payload}],
            temperature=0.2,
            stream=True,
            max_tokens=512
        )

        for chunk in response_stream:
            if chunk.choices[0].delta.content:
                new_text = chunk.choices[0].delta.content
                print(new_text, end="", flush=True)
                full_response += new_text
                sentence_buffer += new_text
                
                if any(char in new_text for char in ['.', '!', '?', '\n']):
                    clean_sentence = sentence_buffer.strip()
                    if len(clean_sentence) > 2:
                        self.speak(clean_sentence)
                    sentence_buffer = ""

        final_fragment = sentence_buffer.strip()
        if len(final_fragment) > 0:
            self.speak(final_fragment)

        print("\n")
        
        # Send a signal to the browser that we finished speaking
        if self.websocket and self.websocket_loop:
            asyncio.run_coroutine_threadsafe(
                self.websocket.send(json.dumps({"speaking": False})),
                self.websocket_loop
            )
            
        return full_response

    def close(self):
        print("Shutting down Iris...")
        self.db_conn.close()