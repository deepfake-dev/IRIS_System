import asyncio
import json
import numpy as np
import warnings
import logging
import os
import openwakeword
from openai import OpenAI

logging.getLogger().setLevel(logging.ERROR)
warnings.filterwarnings("ignore")
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

from openwakeword.model import Model
from faster_whisper import WhisperModel
from kokoro_onnx import Kokoro

# --- NEW RAG IMPORTS ---
import lancedb
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import LanceDB
from flashrank import Ranker, RerankRequest

import google.generativeai as genai

openwakeword.utils.download_models()

class IrisAssistant:
    def __init__(self,
                 wake_word="hey_iris",
                 whisper_size="distil-large-v3",
                 male_voice='am_fenrir',
                 female_voice='bf_isabella',
                 wake_word_path='models/wakeword/hey_iris.onnx',
                 wake_word_threshold = 0.4,
                 stt_threshold = 0.75,
                 lancedb_path="databases/lance_db",                       # Updated to LanceDB
                 table_name="batstateu_rag_nomic",              # Updated to Nomic Table
                 embed_model="nomic-ai/nomic-embed-text-v1.5",  # Updated to Nomic
                 reranker_model="ms-marco-TinyBERT-L-2-v2",     # Updated to FlashRank model
                 rag_top_k=5,
                 initial_retrieval_k=12,
                 websocket=None,
                 websocket_loop=None,
                 iris_ai_port=8001):

        self.wakeword_threshold = wake_word_threshold
        self.male_voice = male_voice
        self.female_voice = female_voice
        self.tts_voice = self.female_voice
        self.rag_top_k = rag_top_k
        self.stt_threshold = stt_threshold
        self.initial_retrieval_k = max(initial_retrieval_k, rag_top_k)
        self.table_name = table_name
        
        self.client = OpenAI(
            base_url=f"http://localhost:{iris_ai_port}/v1",
            api_key="sk-no-key-required"
        )

        self.gemini_model = genai.GenerativeModel('gemini-2.5-flash')

        print(f"Loading Iris Backend (Faster-Whisper {whisper_size})...")

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

        # --- RAG Setup: Nomic Embeddings ---
        print(f"Loading embedding model: {embed_model}")
        self.embedder = HuggingFaceEmbeddings(
            model_name=embed_model,
            model_kwargs={'device': 'cuda', 'trust_remote_code': True}
        )

        # --- RAG Setup: FlashRank ---
        print(f"Loading reranker model: {reranker_model}")
        self.ranker = Ranker(
            model_name=reranker_model, 
            cache_dir=os.path.join(os.getcwd(), "flashrank_cache")
        )

        # --- RAG Setup: LanceDB ---
        print(f"Opening LanceDB database at: {lancedb_path}")
        db_path = os.path.join(os.getcwd(), lancedb_path)
        db = lancedb.connect(db_path)
        self.vector_store = LanceDB(
            connection=db, 
            table_name=self.table_name, 
            embedding=self.embedder
        )
        print(f"Connected to LanceDB table: {self.table_name}")

        self.speaking = False
        self.is_thinking = False
        self.interrupt = False
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

            if score > self.wakeword_threshold:
                print(f"Wake word detected! (score={score:.2f})")

                self.oww_model.reset()

                if self.websocket and self.websocket_loop:
                    asyncio.run_coroutine_threadsafe(
                        self.websocket.send(json.dumps({"wakeword": True})),
                        self.websocket_loop
                    )
                self._wakeword_triggered = True
                self.audio_buffer = np.array([], dtype=np.int16)
                break

    def listen_for_command(self, pcm_chunks):
        audio_flat = np.concatenate(pcm_chunks).flatten().astype(np.float32) / 32768.0

        print("Whisper transcribing...")
        segments, _ = self.whisper.transcribe(
            audio_flat,
            beam_size=5,
            language="en",
            no_speech_threshold=self.stt_threshold,
            log_prob_threshold=-1.0,        
            condition_on_previous_text=False, 
            temperature=0.0,                  
        )
        text = "".join(seg.text for seg in segments).strip()
        print(f"Heard: {text}")

        if self.websocket and self.websocket_loop:
            if text:
                asyncio.run_coroutine_threadsafe(
                    self.websocket.send(json.dumps({"user_query": text})),
                    self.websocket_loop
                )
            asyncio.run_coroutine_threadsafe(
                self.websocket.send(json.dumps({"listening": False})),
                self.websocket_loop
            )

        return text

    def _format_chunk(self, doc: str, meta: dict) -> str:
        source = meta.get("source", meta.get("source_file", "unknown_source"))
        title = meta.get("document_title", source)
        chunk_id = meta.get("chunk_id", meta.get("doc_id", "unknown_chunk"))
        section_path = meta.get("section_path", "")
        page_start = meta.get("page_start")
        page_end = meta.get("page_end")

        header_parts = [f"Source: {source}", f"Title: {title}", f"Chunk: {chunk_id}"]

        if section_path:
            header_parts.append(f"Section: {section_path}")

        if page_start is not None and page_end is not None:
            if page_start == page_end:
                header_parts.append(f"Page: {page_start}")
            else:
                header_parts.append(f"Pages: {page_start}-{page_end}")
        elif page_start is not None:
            header_parts.append(f"Page: {page_start}")

        header = " | ".join(header_parts)
        return f"[{header}]\n{doc}"

    def _retrieve(self, query: str):
        # 1. Initial Retrieval via LanceDB & LangChain
        retrieved_docs = self.vector_store.similarity_search(query, k=self.initial_retrieval_k)

        if not retrieved_docs:
            return "", []

        # 2. Format passages for FlashRank
        passages = [
            {"id": str(i), "text": doc.page_content, "meta": doc.metadata}
            for i, doc in enumerate(retrieved_docs)
        ]

        # 3. Rerank
        rerank_request = RerankRequest(query=query, passages=passages)
        reranked_results = self.ranker.rerank(rerank_request)

        # 4. Extract Top K
        formatted_chunks = []
        retrieval_log = []

        for i, item in enumerate(reranked_results[:self.rag_top_k]):
            meta = item["meta"]
            formatted_chunks.append(self._format_chunk(item["text"], meta))
            
            retrieval_log.append({
                "rank": i + 1,
                "chunk_id": meta.get("chunk_id", meta.get("doc_id", "unknown_chunk")),
                "source": meta.get("source", meta.get("source_file", "unknown_source")),
                "section_path": meta.get("section_path", ""),
                "page_start": meta.get("page_start"),
                "page_end": meta.get("page_end"),
                "distance": "N/A", # FlashRank uses scores instead of vector distance
                "rerank_score": item.get("score", 0.0),
            })

        return "\n\n".join(formatted_chunks), retrieval_log

    def speak(self, text, display_text=None):
        if not text.strip():
            return

        if display_text is None:
            display_text = text

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
                    self.websocket.send(json.dumps({"ai_text_sync": display_text})),
                    self.websocket_loop
                ).result()

                asyncio.run_coroutine_threadsafe(
                    self.websocket.send(wav_bytes),
                    self.websocket_loop
                ).result()

        except Exception as e:
            print(f"TTS Error: {e}")

        finally:
            self.speaking = False

    def chat(self, user_text):
        print(f"📝 You: {user_text}")
        self.is_thinking = True
        self.interrupt = False
        
        context, retrieval_log = self._retrieve(user_text)
        print(f"Retrieved and reranked {len(retrieval_log)} chunks from LanceDB table '{self.table_name}'.")

        for item in retrieval_log:
            print(f"   {item['rank']}. {item['source']} | {item['section_path']} | pages {item['page_start']}-{item['page_end']} | rerank_score={item['rerank_score']:.4f}")

        # STRONGER PROMPT: Forced exactly "CRITICAL_MISSING" for fallback
        iris_context = (
            "You are Iris, an AI assistant for Batangas State University - The National Engineering University - Alangilan Campus.\n"
            "Your output will be read aloud by a Text-to-Speech engine. Answer in a natural, conversational, pure-text format. NO Markdown, bullet points, or emojis.\n\n"
            "CORE KNOWLEDGE RULES:\n"
            "1. STRICT GROUNDING: Use the retrieved campus documents as your absolute and only source of truth.\n"
            "2. MISSING INFORMATION: If the answer is not explicitly found in the retrieved context, you MUST output the exact string \"CRITICAL_MISSING\" and absolutely nothing else. Do not explain or apologize.\n\n"
            "SPECIAL TOOL RULE: DEEPFAKE DETECTION\n"
            "- TRIGGER: ONLY IF the user explicitly asks about \"deepfake detection,\" \"provenance checking,\" or verifying if media is \"AI-generated.\"\n"
            "- ACTION: Include this exact phrasing: \"You can check the authenticity of your media using our Deepfake Detector tool at the following link: http://localhost:4321\""
        )

        combined_payload = (
            f"{iris_context}\n\n"
            f"--- Retrieved Campus Excerpts ---\n{context}\n"
            f"--- End of Excerpts ---\n\n"
            f"User Question: {user_text}"
        )

        print("Iris: ", end="", flush=True)
        
        full_response = ""
        sentence_buffer = ""
        word_count_in_buffer = 0
        WORD_SPEAK_THRESHOLD = 12
        fallback_to_gemini = False

        # --- Helper for streaming, websockets, and TTS ---
        def _process_chunk(new_text, current_response, current_buffer, current_word_count):
            print(new_text, end="", flush=True)
            current_response += new_text
            current_buffer += new_text

            # Mask abbreviations to prevent false sentence boundaries from triggering early
            test_buffer = (current_buffer
                           .replace("Engr. ", "Engr_ ")
                           .replace("Atty. ", "Atty_ ")
                           .replace("Assoc. ", "Assoc_ ")
                           .replace("Prof. ", "Prof_ ")
                           .replace("Dr. ", "Dr_ "))

            split_index = -1
            
            # 1. Try to split at a clean sentence boundary (. ? ! \n)
            for punct in ['. ', '! ', '? ', '\n']:
                idx = test_buffer.rfind(punct)
                if idx != -1:
                    # Split exactly after the punctuation mark
                    split_index = max(split_index, idx + 1)

            # 2. If no sentence boundary, check word limit and split at the last space
            # This guarantees we NEVER cut a token or word in half!
            if split_index == -1 and len(current_buffer.split()) >= WORD_SPEAK_THRESHOLD:
                idx = current_buffer.rfind(' ')
                if idx != -1:
                    split_index = idx

            if split_index != -1:
                complete_words = current_buffer[:split_index].strip()
                leftover = current_buffer[split_index:]
                
                if len(complete_words) > 2:
                    spoken_sentence = (complete_words
                                       .replace("Engr.", "Engineer")
                                       .replace("Atty.", "Attorney")
                                       .replace("Assoc. Prof.", "Associate Professor")
                                       .replace("Assoc.", "Associate")
                                       .replace("Prof.", "Professor")
                                       .replace("Dr.", "Doctor"))
                    self.speak(spoken_sentence, complete_words + " ")
                    
                current_buffer = leftover

            # We no longer need to manually track word count since we calculate it dynamically
            return current_response, current_buffer, 0

        try:
            # 1. ATTEMPT LOCAL QWEN INFERENCE
            response_stream = self.client.chat.completions.create(
                model="qwen3-vl",
                messages=[{"role": "user", "content": combined_payload}],
                stream=True,
                temperature=0.1,
                max_tokens=512,
            )

            # FIX 2: STARTUP BUFFER to prevent failure text from leaking to the UI
            startup_buffer = ""
            streaming_started = False

            for chunk in response_stream:
                if self.interrupt:
                    break

                if chunk.choices[0].delta.content:
                    content = chunk.choices[0].delta.content
                    
                    if not streaming_started:
                        startup_buffer += content
                        
                        # Catch the failure trigger while it's still trapped in the buffer
                        if "CRITICAL" in startup_buffer.upper():
                            fallback_to_gemini = True
                            break # Abort local, switch to Gemini instantly
                        
                        # If we hit 16 safe characters, release the buffer to the UI and continue
                        if len(startup_buffer) > 16:
                            streaming_started = True
                            full_response, sentence_buffer, word_count_in_buffer = _process_chunk(
                                startup_buffer, full_response, sentence_buffer, word_count_in_buffer
                            )
                    else:
                        full_response, sentence_buffer, word_count_in_buffer = _process_chunk(
                            content, full_response, sentence_buffer, word_count_in_buffer
                        )

            # 2. FALLBACK TO GEMINI IF TRIGGERED
            if fallback_to_gemini:
                print("\n[Local context insufficient. Searching with Gemini...]\nIris (via Gemini): ", end="", flush=True)
                
                full_response = "" 
                sentence_buffer = ""
                word_count_in_buffer = 0
                
                gemini_prompt = (
                    f"You are Iris, a helpful voice assistant. Please answer the following question in a conversational, natural tone suitable for text-to-speech. Do not use markdown, bullet points, or emojis.\n\nUser Question: {user_text}"
                )
                
                gemini_stream = self.gemini_model.generate_content(gemini_prompt, stream=True)
                
                for g_chunk in gemini_stream:
                    if self.interrupt:
                        break

                    if g_chunk.text:
                        full_response, sentence_buffer, word_count_in_buffer = _process_chunk(
                            g_chunk.text, full_response, sentence_buffer, word_count_in_buffer
                        )

            # Flush any remaining words in the buffer to TTS
            final_fragment = sentence_buffer.strip()
            if len(final_fragment) > 0:
                spoken_sentence = (final_fragment
                                   .replace("Engr.", "Engineer")
                                   .replace("Atty.", "Attorney")
                                   .replace("Assoc. Prof.", "Associate Professor")
                                   .replace("Assoc.", "Associate")
                                   .replace("Prof.", "Professor")
                                   .replace("Dr.", "Doctor"))
                self.speak(spoken_sentence, final_fragment)

            print("\n")

        except Exception as e:
            print(f"\nError during generation: {e}")

        finally:
            self.oww_model.reset()
            self.is_thinking = False
            if self.websocket and self.websocket_loop:
                asyncio.run_coroutine_threadsafe(
                    self.websocket.send(json.dumps({"speaking": False})),
                    self.websocket_loop
                )

        return full_response

    def close(self):
        print("Shutting down Iris...")