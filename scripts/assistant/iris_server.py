# ==============================================================================
# Copyright (c) 2026 Batangas State University (The National Engineering University)
# Project: IRIS Assistant System
# 
# Attributions:
# - ASR powered by Faster-Whisper (SYSTRAN / Guillaume Klein)
# - Inter-process communication facilitated by the Model Context Protocol (MCP)
# ==============================================================================

import os
import ssl
import json
import time
import asyncio
import argparse
import threading
import websockets
import numpy as np
import random
import logging

from vlm_handler import IrisAssistant
from client_session import ClientSession

# Constants for Speech Filtration
NOISE_WORDS = {"you", "the", "a", "uh", "um", "hmm", "hm", "oh", "ah", "ok", "okay", "yeah", "yes", "no", "bye", "hi", "hey", "thanks", "thank you", "thank", "silence", "music", "applause", "laughter"}
HALLUCINATIONS = {"thank you", "thank you.", "thanks for watching", "thanks for listening", "see you next time", "please subscribe", "you", "bye bye", "thank you very much", "okay", "i'll see you in the next one", "thank you for watching", "thanks for watching."}


class IrisAvatar:
    """Main WebSockets Server bridging kiosk audio streams to the IRIS AI Pipeline."""
    
    def __init__(self, port=7040, api_key="", male_voice="am_fenrir", female_voice="bf_isabella",
                 rag_top_k=5, initial_retrieval_k=12, wakeword_threshold=0.2,
                 stt_silence_threshold=0.75, minimum_audio_level=1500, vlm_alias="gemma-4-e2b-it",
                 local_ip=""):
        
        self.port = port
        self.minimum_audio_level = minimum_audio_level
        self.clients = set()
        
        print("\n[INIT] Loading heavy neural network models into VRAM... Please wait.")
        self.iris = IrisAssistant(
            api_key=api_key,
            male_voice=male_voice,
            female_voice=female_voice,
            rag_top_k=rag_top_k,
            initial_retrieval_k=initial_retrieval_k,
            wake_word_threshold=wakeword_threshold,
            stt_threshold=stt_silence_threshold,
            vlm_alias=vlm_alias,
            local_ip=local_ip
        )
        
        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(target=self._start_server, daemon=True)
        self.thread.start()

    def _start_server(self):
        asyncio.set_event_loop(self.loop)
        self.loop.run_until_complete(self._run_server())

    async def _run_server(self):
        await self.iris._init_mcp_client()

        ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        try:
            ssl_context.load_cert_chain(certfile="cert.pem", keyfile="key.pem")
        except FileNotFoundError:
            print("[WARNING] SSL Certificates not found. Ensure cert.pem and key.pem are present.")

        print(f"✅ IRIS Assistant WebSocket server active on wss://0.0.0.0:{self.port}")
        async with websockets.serve(self._handler, "0.0.0.0", self.port, ssl=ssl_context):
            asyncio.create_task(self._random_blink_loop())
            await asyncio.Future()  # Run forever

    def _process_audio_command(self, session, websocket, chunks_to_process):
        """Threaded executor method handling STT, LLM generation, and cleanup."""
        try:
            # 1. Automatic Speech Recognition (STT)
            command = self.iris.listen_for_command(chunks_to_process)
            
            if command:
                asyncio.run_coroutine_threadsafe(websocket.send(json.dumps({"user_query": command})), self.loop)
            asyncio.run_coroutine_threadsafe(websocket.send(json.dumps({"listening": False})), self.loop)

            # 2. Hallucination and Noise Filtering
            words = command.lower().split() if command else []
            command_clean = command.strip().lower().rstrip(".!?,")
            is_noise = not command or len(words) < 2 or set(words).issubset(NOISE_WORDS) or command_clean in HALLUCINATIONS

            # 3. LLM Generation and Routing
            if not is_noise:
                self.iris.chat(
                    command, 
                    session.on_sentence_ready, 
                    session.check_interrupt,
                    session.kiosk_location
                )

        except Exception as e:
            logging.error(f"Error processing audio command: {e}")
            
        finally:
            # Wait for background TTS to finish, reset state, and signal readiness
            session.tts_queue.join()
            session.interrupt = False
            session.oww_model.reset()
            asyncio.run_coroutine_threadsafe(websocket.send(json.dumps({"speaking": False})), self.loop)
            print("\n✅ Iris is ready for the next command.")

    async def _handler(self, websocket):
        """Primary connection loop handling inbound streaming audio and JSON commands."""
        self.clients.add(websocket)
        session = ClientSession(websocket, self.loop, self.iris)

        try:
            recording_command = False
            command_chunks = []
            silent_chunks = 0
            
            max_silent_chunks = 12 
            max_total_chunks = 40  
            wakeword_step = 1280

            async for message in websocket:
                # --- JSON COMMAND ROUTING ---
                if isinstance(message, str):
                    try:
                        data = json.loads(message)
                        cmd = data.get("command")
                        
                        if cmd == "interrupt":
                            session.interrupt = True
                            session.clear_queue()
                            print("\n[User skipped response. Interrupting AI...]")
                            
                        elif cmd == "set_voice":
                            gender = data.get("gender")
                            session.tts_voice = self.iris.male_voice if gender == "male" else self.iris.female_voice
                            print(f"\n[Client voice switched to {gender}]")
                            
                        elif cmd == "init_kiosk":
                            session.kiosk_location = data.get("location", "Unknown Location")
                            print(f"\n📍 Kiosk connected from: {session.kiosk_location}")
                            
                        elif cmd == "start_listening":
                            if not recording_command:
                                recording_command = True
                                command_chunks.clear()
                                silent_chunks = 0
                                print("\n[Manual Trigger] Listening for command...")
                                await websocket.send(json.dumps({"listening": True}))
                    except json.JSONDecodeError:
                        pass
                    continue

                # --- BINARY AUDIO PROCESSING ---
                if not isinstance(message, bytes): 
                    continue

                chunk = np.frombuffer(message, dtype=np.int16)
                volume = np.sqrt(np.mean(chunk.astype(np.float32) ** 2))

                if not recording_command:
                    # Scan for Wake Word
                    session.audio_buffer = np.concatenate((session.audio_buffer, chunk))
                    triggered = False
                    
                    while len(session.audio_buffer) >= wakeword_step:
                        sub_chunk = session.audio_buffer[:wakeword_step]
                        session.audio_buffer = session.audio_buffer[wakeword_step:]
                        
                        score = session.oww_model.predict(sub_chunk).get(self.iris.wakeword_key, 0.0)
                        if score > self.iris.wakeword_threshold:
                            session.oww_model.reset()
                            triggered = True

                    if triggered:
                        recording_command = True
                        command_chunks.clear()
                        silent_chunks = 0
                        print("\n[Wake Word Triggered] Listening for command...")
                        await websocket.send(json.dumps({"listening": True}))
                else:
                    # Record the Spoken Command
                    command_chunks.append(chunk)
                    print(f"   [Volume: {volume:.0f}]", end="\r")

                    silent_chunks = silent_chunks + 1 if volume < self.minimum_audio_level else 0

                    if silent_chunks > max_silent_chunks or len(command_chunks) > max_total_chunks:
                        print("\nSilence detected. Processing audio...")
                        recording_command = False
                        silent_chunks = 0
                        
                        chunks_to_process = command_chunks[:]
                        command_chunks.clear()

                        # Dispatch to background thread
                        self.loop.run_in_executor(None, self._process_audio_command, session, websocket, chunks_to_process)

        finally:
            self.clients.discard(websocket)
            session.tts_queue.put(None)  # Safely terminate the client's TTS thread

    async def _random_blink_loop(self):
        """Broadcasts passive avatar animations to all connected kiosks."""
        while True:
            await asyncio.sleep(random.uniform(3.0, 7.0))
            self.send_data({"expression": "blink", "intensity": 1.0})
            await asyncio.sleep(0.15)
            self.send_data({"expression": "blink", "intensity": 0.0})

    def send_data(self, data):
        """Utility to broadcast JSON payloads to all clients."""
        if not self.clients: 
            return
        message = json.dumps(data)
        asyncio.run_coroutine_threadsafe(self._broadcast(message), self.loop)

    async def _broadcast(self, message):
        if self.clients:
            await asyncio.gather(*(client.send(message) for client in self.clients))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="IRIS Assistant WebSocket Server")
    parser.add_argument("--port", type=int, default=7040)
    parser.add_argument("--api-key", type=str, default="")
    parser.add_argument("--male-voice", type=str, default="am_fenrir")
    parser.add_argument("--female-voice", type=str, default="bf_isabella")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--initial-retrieval-k", type=int, default=12)
    parser.add_argument("--wakeword-threshold", type=float, default=0.2)
    parser.add_argument("--stt-silence-threshold", type=float, default=0.75)
    parser.add_argument("--minimum-audio-level", type=int, default=1500)
    parser.add_argument("--vlm-alias", type=str, default='gemma-4-e2b-it')
    parser.add_argument("--local-ip", type=str, default="localhost")

    args = parser.parse_args()

    avatar = IrisAvatar(
        port=args.port, 
        api_key=args.api_key,
        male_voice=args.male_voice,
        female_voice=args.female_voice,
        rag_top_k=args.top_k,
        initial_retrieval_k=args.initial_retrieval_k,
        wakeword_threshold=args.wakeword_threshold,
        stt_silence_threshold=args.stt_silence_threshold,
        minimum_audio_level=args.minimum_audio_level,
        vlm_alias=args.vlm_alias,
        local_ip=args.local_ip
    )

    try:
        while True: 
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nShutting down IRIS server...")