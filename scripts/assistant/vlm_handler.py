import collections
import numpy as np
import pyaudio
import sounddevice as sd
import warnings
import logging
import os
import faiss
import sqlite3
import time
from openai import OpenAI

logging.getLogger().setLevel(logging.ERROR)
warnings.filterwarnings("ignore")
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

from openwakeword.model import Model
from faster_whisper import WhisperModel
from sentence_transformers import SentenceTransformer
from kokoro_onnx import Kokoro

import onnxruntime as ort

print(ort.get_available_providers())

class IrisAssistant:
    def __init__(self,
                 wake_word="hey_iris",
                 llama_model="qwen3-vl:2b",
                 whisper_size="small",
                 tts_voice='bf_isabella',
                 wake_word_path='hey_iris.onnx',
                 speaker=None,
                 faiss_path="llama_index/Citizens_Charter_Handbook_2025.faiss",
                 db_path="llama_index/Citizens_Charter_Handbook_2025.db",
                 embed_model="BAAI/bge-small-en-v1.5",
                 rag_top_k=5):
        """
        speaker: any object with a .play(audio, sample_rate) method.
                 If None, falls back to sounddevice.
        """
        self.wake_word = wake_word
        # self.llama_model = llama_model
        self.tts_voice = tts_voice
        self.speaker = speaker
        self.is_running = True
        self.rag_top_k = rag_top_k

        # Audio Config
        self.format = pyaudio.paInt16
        self.channels = 1
        self.rate = 16000
        self.chunk = 1280
        self.silence_threshold = 3000
        self.silence_duration = 4.0
        self.oww_flush_seconds = 3.0

        self.client = OpenAI(base_url="http://localhost:8001/v1", api_key="sk-no-key-required")

        print(f"Loading Iris ({llama_model} / {whisper_size})...")

        # --- 1. Wake Word ---
        ww_path = wake_word_path if wake_word_path else f"{wake_word}.onnx"
        self.wakeword_key = os.path.splitext(os.path.basename(ww_path))[0]

        try:
            self.oww_model = Model(wakeword_models=[ww_path], inference_framework="onnx")
        except Exception as e:
            print(f"⚠️ Could not load {ww_path}: {e}")
            print("   Falling back to hey_jarvis...")
            self.oww_model = Model(wakeword_models=["hey_jarvis"], inference_framework="onnx")
            self.wakeword_key = "hey_jarvis"

        print(f"   Wake word key: '{self.wakeword_key}'")

        # --- 2. Whisper ---
        self.whisper = WhisperModel(whisper_size, device="cuda", compute_type="int8")

        # --- 3. TTS ---
        # from kokoro import KPipeline
        # self.tts = KPipeline(lang_code='a', device="cuda", repo_id='hexgrad/Kokoro-82M')
        self.tts = Kokoro("kokoro-v1.0.onnx", "voices-v1.0.bin")


        # --- 4. Audio Stream ---
        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(
            format=self.format, channels=self.channels, rate=self.rate,
            input=True, frames_per_buffer=self.chunk
        )
        self.buffer = collections.deque(maxlen=int(self.rate / self.chunk))

        # --- 5. RAG Assets ---
        print("Loading RAG assets...")
        self.embed_model = SentenceTransformer(embed_model)
        self.faiss_index = faiss.read_index(faiss_path)
        self.db_conn = sqlite3.connect(db_path, check_same_thread=False)
        print("✅ RAG ready.")

        print("✅ Iris is ready.")

        self.speaking = False

    # ------------------------------------------------------------------ #
    #  RAG                                                                 #
    # ------------------------------------------------------------------ #

    def _retrieve(self, query):
        """Embed the query, search FAISS, return top-k chunk texts."""
        q_vec = self.embed_model.encode(
            [query], normalize_embeddings=True
        ).astype("float32")

        _, ids = self.faiss_index.search(q_vec, self.rag_top_k)

        cur = self.db_conn.cursor()
        placeholders = ",".join("?" * len(ids[0]))
        cur.execute(
            f"SELECT text FROM chunks WHERE row_id IN ({placeholders})",
            ids[0].tolist()
        )
        rows = cur.fetchall()
        return "\n\n".join(row[0] for row in rows)

    # ------------------------------------------------------------------ #
    #  TTS                                                                 #
    # ------------------------------------------------------------------ #

    def speak(self, text):
        self.speaking = True
        if not text.strip():
            return
        samples, sample_rate = self.tts.create(
            text, voice=self.tts_voice, speed=1.0, lang="en-us"
        )

        silence_padding = np.zeros(int(sample_rate * 0.3), dtype=samples.dtype)
        samples_padded = np.concatenate((samples, silence_padding))

        if self.speaker:
            sd.play(samples_padded, sample_rate)
            sd.wait()
        self.speaking = False
        # generator = self.tts(text, voice=self.tts_voice, speed=1.0, split_pattern=r'\n+')
        # for _, _, audio in generator:
        #     if self.speaker:
        #         self.speaker.play(audio, 24000)
        #     else:
        #         import sounddevice as sd
        #         sd.play(audio, 24000)
        #         sd.wait()

    # ------------------------------------------------------------------ #
    #  LISTENING                                                           #
    # ------------------------------------------------------------------ #

    def listen_for_command(self):
        recorded_audio = list(self.buffer)
        silent_chunks = 0
        max_silent_chunks = int(self.silence_duration * (self.rate / self.chunk))

        self.speaker.animate_listen()

        print("👂 Listening for command...")

        while True:
            data = self.stream.read(self.chunk, exception_on_overflow=False)
            audio_chunk = np.frombuffer(data, dtype=np.int16)
            recorded_audio.append(audio_chunk)

            volume = np.sqrt(np.mean(audio_chunk.astype(np.float32) ** 2))
            if volume < self.silence_threshold:
                silent_chunks += 1
            else:
                silent_chunks = 0

            if silent_chunks > max_silent_chunks:
                break

        audio_flat = np.concatenate(recorded_audio).flatten().astype(np.float32) / 32768.0
        segments, _ = self.whisper.transcribe(audio_flat, beam_size=5, language="en")
        self.speaker.animate_unlisten()
        return "".join(seg.text for seg in segments).strip()

    # ------------------------------------------------------------------ #
    #  CHAT                                                                #
    # ------------------------------------------------------------------ #

    def chat(self, user_text):
        print(f"📝 You: {user_text}")

        # 1. Retrieve relevant chunks from the Citizens Charter
        context = self._retrieve(user_text)
        print(f"📚 Retrieved {self.rag_top_k} chunks from Citizens Charter.")

        # 2. Build prompt
        iris_context = (
            "You are Iris, an AI assistant for Batangas State University - "
            "The National Engineering University - Alangilan Campus. "
            "Answer in pure text form, like it will be read as is, and do not format to Markdown."
            "Your birthday is February 30, 1777. You never use emojis or emoticons. "
            "You answer only specific information about yourself and Batangas State University. "
            "Batangas State University is located in Golden Country Homes, Alangilan, "
            "Batangas City, Batangas Province, Philippines. "
            "By virtue of Republic Act No. 11694, Batangas State University is designated "
            "as The National Engineering University in the Philippines. "
            "The President of Batangas State University (BSU) is Dr. Tirso A. Ronquillo. "
        )

        combined_payload = (
            f"{iris_context}\n\n"
            f"Use the following excerpts from the Citizens Charter Handbook to help answer "
            f"the question. If the answer is not found in the excerpts, rely on your general "
            f"knowledge about Batangas State University.\n\n"
            f"--- Citizens Charter Excerpts ---\n{context}\n"
            f"--- End of Excerpts ---\n\n"
            f"User Question: {user_text}"
        )

        # 3. Stream response from Ollama
        print("🤖 Iris: ", end="", flush=True)
        full_response = ""
        sentence_buffer = ""

        # This looks exactly like an OpenAI API call, but runs 100% locally
        response_stream = self.client.chat.completions.create(
            model="qwen3-vl", # The name doesn't matter, llama.cpp routes it automatically
            messages=[{"role": "user", "content": combined_payload}],
            temperature=0.2,
            extra_body={"thinking": False},
            stream=True,
            max_tokens=512
        )

        for chunk in response_stream:
            # Check if the chunk contains text (OpenAI streams sometimes send empty chunks)
            if chunk.choices[0].delta.content:
                new_text = chunk.choices[0].delta.content
                print(new_text, end="", flush=True)
                full_response += new_text
                sentence_buffer += new_text
                
                # Speak as soon as a full sentence is formed
                if any(char in new_text for char in ['.', '!', '?', '\n']):
                    clean_sentence = sentence_buffer.strip()
                    if len(clean_sentence) > 2:
                        self.speak(clean_sentence)
                    sentence_buffer = ""

        # Catch any leftover text
        final_fragment = sentence_buffer.strip()
        if len(final_fragment) > 0:
            self.speak(final_fragment)

        print("\n")
        return full_response

    # ------------------------------------------------------------------ #
    #  WAKE WORD FLUSH                                                     #
    # ------------------------------------------------------------------ #

    def _flush_wakeword(self):
        available = self.stream.get_read_available()
        while available >= self.chunk:
            self.stream.read(self.chunk, exception_on_overflow=False)
            available = self.stream.get_read_available()

        silence = np.zeros(self.chunk, dtype=np.int16)
        for _ in range(int((self.rate / self.chunk) * self.oww_flush_seconds)):
            self.oww_model.predict(silence)

        if self.wakeword_key in self.oww_model.prediction_buffer:
            self.oww_model.prediction_buffer[self.wakeword_key].clear()

    # ------------------------------------------------------------------ #
    #  MAIN LOOP                                                           #
    # ------------------------------------------------------------------ #

    def run_forever(self):
        print(f"Waiting for '{self.wake_word}'...")

        try:
            while self.is_running:
                data = self.stream.read(self.chunk, exception_on_overflow=False)
                audio_int16 = np.frombuffer(data, dtype=np.int16)
                self.buffer.append(audio_int16)

                prediction = self.oww_model.predict(audio_int16)
                score = prediction.get(self.wakeword_key, 0.0)

                if score > 0.5:
                    print(f"⚡ Wake word detected! (score={score:.2f})")

                    command = self.listen_for_command()

                    if command:
                        self.chat(command)
                    else:
                        print("⚠️ No speech detected.")

                    self._flush_wakeword()
                    self.buffer.clear()
                    print(f"Waiting for '{self.wake_word}'...")

        except KeyboardInterrupt:
            self.close()

    # ------------------------------------------------------------------ #
    #  CLEANUP                                                             #
    # ------------------------------------------------------------------ #

    def close(self):
        print("Shutting down Iris...")
        self.stream.stop_stream()
        self.stream.close()
        self.p.terminate()
        self.db_conn.close()