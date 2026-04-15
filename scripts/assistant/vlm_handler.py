# ==============================================================================
# Copyright (c) 2026 Batangas State University (The National Engineering University)
# Project: IRIS Assistant System
# 
# Attributions:
# - Wake Word: OpenWakeWord (David Scripka)
# - STT: Faster-Whisper (SYSTRAN / Guillaume Klein)
# - TTS: Kokoro-ONNX (StyleTTS2 architecture)
# - RAG: LanceDB & HuggingFace (Nomic Embeddings)
# - Reranking: FlashRank (Prithivi Da)
# - Tooling: Model Context Protocol (Anthropic)
# ==============================================================================

import os
import sys
import json
import logging
import asyncio
import warnings
import numpy as np
from contextlib import AsyncExitStack

# MCP Client Imports
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# AI / ML Imports
import openwakeword
from openwakeword.model import Model
from faster_whisper import WhisperModel
from kokoro_onnx import Kokoro
from openai import OpenAI
import lancedb
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import LanceDB
from flashrank import Ranker, RerankRequest

# Local Utilities
from text_utils import get_last_valid_split, format_spoken_text

# System Configuration
logging.getLogger().setLevel(logging.ERROR)
warnings.filterwarnings("ignore")
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
openwakeword.utils.download_models()


class IrisAssistant:
    def __init__(self,
                 wake_word="hey_iris",
                 whisper_size="large-v3",
                 male_voice='am_fenrir',
                 female_voice='bf_isabella',
                 wake_word_path='models/wakeword/hey_iris.onnx',
                 wake_word_threshold=0.2,
                 stt_threshold=0.75,
                 lancedb_path="databases/rag_db",
                 table_name="batstateu_info",
                 embed_model="nomic-ai/nomic-embed-text-v1.5",
                 reranker_model="ms-marco-TinyBERT-L-2-v2",
                 rag_top_k=5,
                 initial_retrieval_k=12,
                 iris_ai_port=8001,
                 api_key="",
                 vlm_alias="gemma-4-e2b-it",
                 local_ip="localhost"):

        self.wakeword_threshold = wake_word_threshold
        self.wake_word_path = wake_word_path
        self.wakeword_key = os.path.splitext(os.path.basename(wake_word_path))[0]
        self.male_voice = male_voice
        self.female_voice = female_voice
        self.rag_top_k = rag_top_k
        self.stt_threshold = stt_threshold
        self.initial_retrieval_k = max(initial_retrieval_k, rag_top_k)
        self.table_name = table_name
        self.vlm_name = vlm_alias
        self.local_ip = local_ip
        
        # 1. Core Models Setup
        self.client = OpenAI(base_url=f"http://{local_ip}:{iris_ai_port}/v1", api_key="sk-no-key-required")
        self.whisper = WhisperModel(whisper_size, device="cuda", compute_type="int8")
        self.tts = Kokoro("models/tts/kokoro-v1.0.onnx", "models/tts/voices-v1.0.bin")

        try:
            self.oww_model = Model(wakeword_models=[wake_word_path], inference_framework="onnx")
        except Exception as e:
            logging.warning(f"Failed to load specific wake word, falling back: {e}")
            self.oww_model = Model(wakeword_models=["hey_jarvis"], inference_framework="onnx")

        # 2. RAG Setup
        self.embedder = HuggingFaceEmbeddings(
            model_name=embed_model, 
            model_kwargs={'device': 'cuda', 'trust_remote_code': True}
        )
        self.ranker = Ranker(model_name=reranker_model, cache_dir=os.path.join(os.getcwd(), "flashrank_cache"))
        db = lancedb.connect(os.path.join(os.getcwd(), lancedb_path))
        self.vector_store = LanceDB(connection=db, table_name=self.table_name, embedding=self.embedder)
        
        # 3. MCP Client Session State
        self.mcp_session = None
        self.mcp_tools = []
        self.mcp_loop = None

    async def _init_mcp_client(self):
        """Connects to the local MCP server and fetches available tools."""
        self.mcp_loop = asyncio.get_running_loop()
        server_script = os.path.join(os.path.dirname(__file__), "mcp_server.py")
        
        server_params = StdioServerParameters(
            command=sys.executable, 
            args=[server_script],
            env=os.environ.copy()
        )
        
        self.exit_stack = AsyncExitStack()
        stdio_transport = await self.exit_stack.enter_async_context(stdio_client(server_params))
        self.read, self.write = stdio_transport
        self.mcp_session = await self.exit_stack.enter_async_context(ClientSession(self.read, self.write))
        await self.mcp_session.initialize()
        
        mcp_tools_response = await self.mcp_session.list_tools()
        for tool in mcp_tools_response.tools:
            self.mcp_tools.append({
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.inputSchema
                }
            })
        print(f"✅ Loaded MCP Tools: {[t['function']['name'] for t in self.mcp_tools]}")

    def listen_for_command(self, pcm_chunks):
        """Transcribes raw audio arrays into text using Faster-Whisper."""
        audio_flat = np.concatenate(pcm_chunks).flatten().astype(np.float32) / 32768.0
        segments, _ = self.whisper.transcribe(
            audio_flat, beam_size=5, language="en", no_speech_threshold=self.stt_threshold,
            log_prob_threshold=-1.0, condition_on_previous_text=False, temperature=0.0,
        )
        return "".join(seg.text for seg in segments).strip()

    def _retrieve(self, query: str) -> str:
        """Executes Hybrid Vector Search + TinyBERT Reranking via LanceDB."""
        try:
            query_vector = self.embedder.embed_query(f"search_query: {query}")
            results_df = self.vector_store._table.search(query_vector).limit(self.initial_retrieval_k).to_pandas()
            
            if results_df.empty: 
                return ""
                
            passages = [
                {
                    "id": str(i), 
                    "text": row["text"], 
                    "meta": {"doc_id": row.get("doc_id", "unknown")}
                } 
                for i, row in results_df.iterrows()
            ]
            
            reranked = self.ranker.rerank(RerankRequest(query=query, passages=passages))
            top_results = [item["text"] for item in reranked[:self.rag_top_k]]
            
            # Debugging Output
            print("\n" + "="*50)
            print(f"🔍 RETRIEVED {len(top_results)} CHUNKS FOR QUERY: '{query}'")
            print("="*50)
            for i, text in enumerate(top_results):
                print(f"\n--- CHUNK {i+1} ---\n{text}")
            print("\n" + "="*50 + "\n")
                
            return "\n\n".join(top_results)
            
        except Exception as e:
            logging.error(f"[RETRIEVAL ERROR] {e}")
            return ""

    def chat(self, user_text, on_sentence_ready, check_interrupt, current_location="Unknown Location"):
        """Agentic chat loop with MCP Tool support and RAG integration."""
        context = self._retrieve(user_text)
        
        # system_prompt = (
        #     "You are Iris, an AI assistant for Batangas State University.\n\n"
        #     "--- TOOL USAGE INSTRUCTIONS (CRITICAL) ---\n"
        #     "1. You MUST read the 'Retrieved Campus Database Context' below FIRST.\n"
        #     "2. If you do not know the answer, call the 'ask_gemini' tool.\n"
        #     "2. If the context contains the answer, you MUST use it to answer the user. Do NOT call any tools.\n"
        #     "3. ONLY use the 'ask_gemini' tool if you have thoroughly read the context and the answer is completely missing.\n"
        #     "4. ONLY use the 'open_deepfake_detector' tool if the user is asking for a deepfake check or a provenance check for their videos.\n\n"
        #     "--- FINAL SPOKEN RESPONSE FORMAT ---\n"
        #     "Once you have your final answer, speak it naturally in pure-text. Do NOT use Markdown, asterisks, or bullet points.\n\n"
        #     f"--- Retrieved Campus Database Context ---\n{context}\n"
        # )

        system_prompt = system_prompt = (
            "You are Iris, an AI assistant for Batangas State University.\n\n"
            "--- EXECUTION LOGIC (CRITICAL) ---\n"
            "Step 1: Read the 'Retrieved Campus Database Context' below.\n"
            "Step 2: Evaluate if the context contains the exact answer to the user's query.\n"
            "Step 3: Apply the following conditional logic:\n"
            "  - IF THE ANSWER IS IN THE CONTEXT: Answer the user directly. Do NOT call any tools.\n"
            "  - IF THE ANSWER IS MISSING: You MUST instantly call the 'ask_gemini' tool. NEVER say 'I do not have enough information', NEVER apologize, and NEVER generate spoken text. ONLY output the tool call.\n"
            "  - IF THE USER ASKS ABOUT DEEPFAKES/PROVENANCE: Instantly call the 'open_deepfake_detector' tool.\n\n"
            "--- FINAL SPOKEN RESPONSE FORMAT ---\n"
            "When you are ready to give your final answer (either from the context or after receiving a tool's output), you must speak naturally in pure conversational text.\n"
            "You are connected to a Text-to-Speech engine. You MUST NOT use any Markdown, asterisks, bolding, bullet points, or special characters. Write numbers as words if necessary.\n\n"
            # "IF you are NOT SURE about the answer, try using 'ask_gemini' tool."
            f"--- Retrieved Campus Database Context ---\n{context}\n"
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_text}
        ]

        # Phase 1: Tool Calling Evaluation (Non-streaming)
        response = self.client.chat.completions.create(
            model=self.vlm_name,
            messages=messages,
            tools=self.mcp_tools,
            tool_choice="auto",
            temperature=0.1
        )

        message = response.choices[0].message
        messages.append(message)

        # Phase 2: Execute MCP Tools if requested
        if message.tool_calls:
            for tool_call in message.tool_calls:
                func_name = tool_call.function.name
                args = json.loads(tool_call.function.arguments)
                
                if func_name == "get_kiosk_location":
                    args["current_location"] = current_location

                print(f"\n[Agent executing tool: {func_name}]")
                try:
                    # Safely await the async tool call from the synchronous thread
                    future = asyncio.run_coroutine_threadsafe(
                        self.mcp_session.call_tool(func_name, arguments=args),
                        self.mcp_loop
                    )
                    result = future.result() 
                    tool_content = result.content[0].text
                except Exception as e:
                    tool_content = f"Error executing tool: {str(e)}"

                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": tool_content
                })

            # --- THE WAKE-UP NUDGE ---
            # This forces the local LLM to speak the tool's result out loud instead of returning an empty string.
            messages.append({
                "role": "user",
                "content": "The tool successfully executed and returned the result above. Please summarize what you just did for me naturally."
            })

            # Stream the final answer after tool execution
            stream_response = self.client.chat.completions.create(
                model=self.vlm_name,
                messages=messages,
                stream=True
            )
        else:
            # Re-run as a stream if no tools were needed
            messages.pop() 
            stream_response = self.client.chat.completions.create(
                model=self.vlm_name,
                messages=messages,
                stream=True
            )

        # Phase 3: Sentence Chunking & TTS Streaming
        self._stream_to_tts(stream_response, on_sentence_ready, check_interrupt)
    
    def _stream_to_tts(self, stream_response, on_sentence_ready, check_interrupt):
        """Consumes the LLM token stream and chunks it into sentences for the TTS engine."""
        full_response, sentence_buffer = "", ""

        for chunk in stream_response:
            if check_interrupt(): break
            if chunk.choices[0].delta.content:
                content = chunk.choices[0].delta.content
                print(content, end="", flush=True)
                full_response += content
                sentence_buffer += content

                split_index = get_last_valid_split(sentence_buffer)

                if split_index != -1:
                    complete_words = sentence_buffer[:split_index].strip()
                    if len(complete_words) > 2:
                        spoken = format_spoken_text(complete_words)
                        on_sentence_ready(spoken, complete_words + " ")
                    sentence_buffer = sentence_buffer[split_index:]

        final_fragment = sentence_buffer.strip()
        if final_fragment and not check_interrupt():
            spoken = format_spoken_text(final_fragment)
            on_sentence_ready(spoken, final_fragment)