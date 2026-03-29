import numpy as np
import json
import threading
import asyncio
import websockets
import random
import time

from vlm_handler import IrisAssistant

class IrisAvatar:
    speaking = False
    
    def __init__(self, port=8080):
        self.port = port
        self.clients = set()
        self.current_pose = {}
        self.iris = IrisAssistant()
        print(f"IRIS Assistant server starting on ws://localhost:{port}")
        
        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(target=self._start_server, daemon=True)
        self.thread.start()

    def _start_server(self):
        asyncio.set_event_loop(self.loop)
        self.loop.run_until_complete(self._run_server())

    async def _run_server(self):
        async with websockets.serve(self._handler, "0.0.0.0", self.port):
            asyncio.create_task(self._random_blink_loop())
            await asyncio.Future()

    async def _handler(self, websocket):
        self.clients.add(websocket)
        self.iris.websocket = websocket
        self.iris.websocket_loop = self.loop
        self.iris._wakeword_triggered = False

        try:
            recording_command = False
            command_chunks = []
            silent_chunks = 0

            # Browser sends ~4096 samples per chunk (about 0.25 seconds)
            max_silent_chunks = 12 # ~3 seconds of silence needed to stop
            max_total_chunks = 40  # ~10 seconds maximum recording time failsafe

            async for message in websocket:
                if isinstance(message, str):
                    try:
                        data = json.loads(message)
                        if data.get("command") == "interrupt":
                            self.iris.interrupt = True
                            print("\n[User skipped response. Interrupting AI...]")
                        elif data.get("command") == "set_voice":
                            gender = data.get("gender")
                            print(gender)
                            if gender == "male":
                                self.iris.tts_voice = self.iris.male_voice
                            else:
                                self.iris.tts_voice = self.iris.female_voice
                                
                            print(f"\n[Voice switched to {gender} ({self.iris.tts_voice})]")
                    except:
                        pass
                    continue

                if not isinstance(message, bytes):
                    continue

                chunk = np.frombuffer(message, dtype=np.int16)
                volume = np.sqrt(np.mean(chunk.astype(np.float32) ** 2))

                if not recording_command:
                    # ── STRICT WAKE WORD MODE ONLY ──────────────────────────────
                    self.iris.process_audio_chunk(chunk)
                    
                    if self.iris._wakeword_triggered:
                        recording_command = True
                        command_chunks = []
                        silent_chunks = 0
                        self.iris._wakeword_triggered = False
                        print("\nListening for command...")
                        await websocket.send(json.dumps({"listening": True}))
                else:
                    # ── RECORDING USER COMMAND ─────────────────────────────────
                    command_chunks.append(chunk)
                    
                    print(f"   [Volume: {volume:.0f}]", end="\r")

                    if volume < 1500:
                        silent_chunks += 1
                    else:
                        silent_chunks = 0

                    if silent_chunks > max_silent_chunks or len(command_chunks) > max_total_chunks:
                        print("\nSilence detected. Processing audio...")
                        recording_command = False
                        self.iris.is_thinking = True
                        silent_chunks = 0
                        chunks_to_process = command_chunks[:]
                        command_chunks = []

                        def respond():
                            try:
                                command = self.iris.listen_for_command(chunks_to_process)

                                # Whisper noise/hallucination filter
                                NOISE_WORDS = {
                                    "you", "the", "a", "uh", "um", "hmm", "hm",
                                    "oh", "ah", "ok", "okay", "yeah", "yes", "no",
                                    "bye", "hi", "hey", "thanks", "thank you",
                                    "thank", "silence", "music", "applause", "laughter",
                                }
                                HALLUCINATIONS = {
                                    "thank you", "thank you.", "thanks for watching",
                                    "thanks for listening", "see you next time",
                                    "please subscribe", "you", "bye bye",
                                    "thank you very much", "okay",
                                    "i'll see you in the next one",
                                    "thank you for watching", "thanks for watching.",
                                }

                                words = command.lower().split() if command else []
                                command_clean = command.strip().lower().rstrip(".!?,")

                                is_noise = (
                                    not command
                                    or len(words) < 2
                                    or set(words).issubset(NOISE_WORDS)
                                    or command_clean in HALLUCINATIONS
                                )

                                if is_noise:
                                    print(f"Ignored likely noise: '{command}' — waiting for wake word.")
                                    if self.iris.websocket and self.iris.websocket_loop:
                                        asyncio.run_coroutine_threadsafe(
                                            self.iris.websocket.send(json.dumps({"listening": False})),
                                            self.iris.websocket_loop
                                        )
                                else:
                                    # Send directly to the AI
                                    self.iris.chat(command)
                                    
                            finally:
                                self.iris.oww_model.reset()
                                self.iris.is_thinking = False
                                print("\n✅ Iris is ready for the next command.")

                        # Run the response logic in the background
                        self.loop.run_in_executor(None, respond)

        finally:
            self.iris.websocket = None
            self.clients.discard(websocket)

    async def _random_blink_loop(self):
        while True:
            await asyncio.sleep(random.uniform(3.0, 7.0))

            self.send_data({"expression": "blink", "intensity": 1.0})
            await asyncio.sleep(0.15)
            self.send_data({"expression": "blink", "intensity": 0.0})

    def send_data(self, data):
        if not self.clients:
            return
        message = json.dumps(data)
        asyncio.run_coroutine_threadsafe(self._broadcast(message), self.loop)

    async def _broadcast(self, message):
        if self.clients:
            await asyncio.gather(*(client.send(message) for client in self.clients))
        
if __name__ == "__main__":
    avatar = IrisAvatar(port=8080)
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Shutting down.")