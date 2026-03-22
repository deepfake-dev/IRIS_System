# pip install chromadb sentence-transformers
import asyncio
import json
import numpy as np
import warnings
import logging
import os
import chromadb
import openwakeword
from openai import OpenAI

logging.getLogger().setLevel(logging.ERROR)
warnings.filterwarnings("ignore")
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

from openwakeword.model import Model
from faster_whisper import WhisperModel
from sentence_transformers import SentenceTransformer
from kokoro_onnx import Kokoro

openwakeword.utils.download_models()

class IrisAssistant:
    def __init__(self,
                 wake_word="hey_iris",
                 whisper_size="small",
                 tts_voice='bf_isabella',
                 wake_word_path='models/wakeword/hey_iris.onnx',
                 chroma_path="databases/chroma_db",
                 collection_name="batstateu_rag",
                 embed_model="BAAI/bge-small-en-v1.5",
                 rag_top_k=5,
                 websocket=None,
                 websocket_loop=None):

        self.tts_voice = tts_voice
        self.rag_top_k = rag_top_k
        self.client = OpenAI(
            base_url="http://localhost:8001/v1",
            api_key="sk-no-key-required"
        )

        print(f"Loading Iris Backend (Whisper {whisper_size})...")

        # --- Wake Word ---
        try:
            self.oww_model = Model(
                wakeword_models=[wake_word_path],
                inference_framework="onnx"
            )
            self.wakeword_key = os.path.splitext(
                os.path.basename(wake_word_path)
            )[0]
            print(f"✅ Loaded custom wake word: {self.wakeword_key}")
        except Exception as e:
            print(f"⚠️ Could not load {wake_word_path}: {e}")
            print("   Falling back to hey_jarvis...")
            self.oww_model = Model(
                wakeword_models=["hey_jarvis"],
                inference_framework="onnx"
            )
            self.wakeword_key = "hey_jarvis"

        # --- Whisper ---
        self.whisper = WhisperModel(
            whisper_size,
            device="cuda",
            compute_type="int8"
        )

        self.websocket = websocket
        self.websocket_loop = websocket_loop

        # --- TTS ---
        self.tts = Kokoro(
            "models/tts/kokoro-v1.0.onnx",
            "models/tts/voices-v1.0.bin"
        )

        # --- RAG / Chroma ---
        print(f"Loading embedding model: {embed_model}")
        self.embed_model = SentenceTransformer(embed_model)

        print(f"Opening Chroma database at: {chroma_path}")
        self.chroma_client = chromadb.PersistentClient(path=chroma_path)
        self.collection = self.chroma_client.get_collection(collection_name)
        print(f"✅ Connected to Chroma collection: {collection_name}")
        print(f"📚 Total chunks in collection: {self.collection.count()}")

        self.speaking = False
        self.is_thinking = False
        self._wakeword_triggered = False
        self.audio_buffer = np.array([], dtype=np.int16)

    def process_audio_chunk(self, pcm_int16: np.ndarray):
        if self.speaking or getattr(self, 'is_thinking', False):
            self.audio_buffer = np.array([], dtype=np.int16)
            return

        self.audio_buffer = np.concatenate((self.audio_buffer, pcm_int16))
        step = 1280

        while len(self.audio_buffer) >= step:
            sub_chunk = self.audio_buffer[:step]
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
                self.audio_buffer = np.array([], dtype=np.int16)
                break

    def listen_for_command(self, pcm_chunks):
        if self.websocket and self.websocket_loop:
            asyncio.run_coroutine_threadsafe(
                self.websocket.send(json.dumps({"listening": True})),
                self.websocket_loop
            )

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
        q_vec = self.embed_model.encode(
            [query],
            normalize_embeddings=True
        ).tolist()

        results = self.collection.query(
            query_embeddings=q_vec,
            n_results=self.rag_top_k
        )

        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]

        formatted_chunks = []
        for i, doc in enumerate(docs):
            meta = metas[i] if i < len(metas) else {}
            source = meta.get("source_file", "unknown_source")
            doc_id = meta.get("doc_id", "unknown_doc")

            formatted_chunks.append(
                f"[Source: {source} | Doc: {doc_id}]\n{doc}"
            )

        return "\n\n".join(formatted_chunks)

    def speak(self, text):
        if not text.strip():
            return

        self.speaking = True

        try:
            samples, sample_rate = self.tts.create(
                text,
                voice=self.tts_voice,
                speed=1.0,
                lang="en-us"
            )

            samples = np.clip(samples, -1.0, 1.0)

            import io
            import wave

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

        self.is_thinking = True
        context = self._retrieve(user_text)
        print(f"📚 Retrieved {self.rag_top_k} chunks from Chroma.")

        iris_context = (
            "You are Iris, an AI assistant for Batangas State University - "
            "The National Engineering University - Alangilan Campus. "
            "Answer in pure text form, as if it will be spoken aloud, and do not use Markdown. "
            "You never use emojis or emoticons. "
            "Use the retrieved campus documents as your main source of truth. "
            "If the answer is not found in the retrieved context, say that you are not sure based on the available campus documents. "
            "Do not invent office names, procedures, schedules, requirements, or fees."
        )

        combined_payload = (
            f"{iris_context}\n\n"
            f"Use the following retrieved campus excerpts to answer the user's question.\n\n"
            f"--- Retrieved Campus Excerpts ---\n{context}\n"
            f"--- End of Excerpts ---\n\n"
            f"User Question: {user_text}"
        )

        print("🤖 Iris: ", end="", flush=True)
        full_response = ""
        sentence_buffer = ""

        try:
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

        finally:
            self.is_thinking = False

            if self.websocket and self.websocket_loop:
                asyncio.run_coroutine_threadsafe(
                    self.websocket.send(json.dumps({"speaking": False})),
                    self.websocket_loop
                )

        return full_response

    def close(self):
        print("Shutting down Iris...")