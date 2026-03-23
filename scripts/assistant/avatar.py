import numpy as np
import json
import threading
import asyncio
import websockets
import random
import time

from vlm_handler import IrisAssistant

class IrisAvatar:
    wave_animating = False
    speaking = False
    
    def __init__(self, port=8080):
        self.port = port
        self.clients = set()
        self.current_pose = {}
        self.iris = IrisAssistant()
        print(f"📡 Avatar WebSocket server starting on ws://localhost:{port}")
        
        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(target=self._start_server, daemon=True)
        self.thread.start()

    def _start_server(self):
        asyncio.set_event_loop(self.loop)
        self.loop.run_until_complete(self._run_server())

    async def _run_server(self):
        async with websockets.serve(self._handler, "0.0.0.0", self.port):
            asyncio.create_task(self._random_blink_loop())
            asyncio.create_task(self._random_idle_loop())
            await asyncio.Future()

    async def _handler(self, websocket):
        self.clients.add(websocket)
        self.iris.websocket = websocket
        self.iris.websocket_loop = self.loop
        self.iris._wakeword_triggered = False

        try:
            neutral = {'leftUpperArm': {'z': -1.2}, 'rightUpperArm': {'z': 1.2}}
            for bone, rot in neutral.items():
                self.current_pose[bone] = rot
                await websocket.send(json.dumps({'bone': bone, 'rotation': rot}))

            recording_command = False
            command_chunks = []
            silent_chunks = 0
            
            # Browser sends ~4096 samples per chunk (about 0.25 seconds)
            max_silent_chunks = 12 # ~3 seconds of silence needed to stop
            max_total_chunks = 40  # ~10 seconds maximum recording time failsafe

            async for message in websocket:
                if not isinstance(message, bytes):
                    continue

                chunk = np.frombuffer(message, dtype=np.int16)

                if not recording_command:
                    self.iris.process_audio_chunk(chunk)
                    if self.iris._wakeword_triggered:
                        recording_command = True
                        command_chunks = []
                        silent_chunks = 0
                        self.iris._wakeword_triggered = False
                        print("\n🎤 Listening for command...")
                        # Fire listening indicator immediately on wake word
                        await websocket.send(json.dumps({"listening": True}))
                else:
                    command_chunks.append(chunk)
                    
                    # Calculate Root Mean Square (RMS) volume
                    volume = np.sqrt(np.mean(chunk.astype(np.float32) ** 2))
                    
                    # DEBUG: This prints the volume so you can see your room's background noise level
                    # If this prints 2000 when you are quiet, change the 1500 below to 2500.
                    print(f"   [Volume: {volume:.0f}]", end="\r")

                    if volume < 1500: # <-- TUNING KNOB: Increase this if it gets stuck
                        silent_chunks += 1
                    else:
                        silent_chunks = 0

                    # Stop if we hit enough silence OR if we've been recording for too long
                    if silent_chunks > max_silent_chunks or len(command_chunks) > max_total_chunks:
                        print("\n🛑 Silence detected. Processing audio...")
                        recording_command = False
                        
                        # 🔒 LOCK the AI's ears so it doesn't get confused by background noise
                        self.iris.is_thinking = True 
                        
                        silent_chunks = 0
                        chunks_to_process = command_chunks[:]
                        command_chunks = []

                        def respond():
                            try:
                                command = self.iris.listen_for_command(chunks_to_process)

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
                                GOODBYE_PHRASES = {
                                    "goodbye", "good bye", "bye", "bye bye",
                                    "see you", "see you later", "see ya",
                                    "that's all", "thats all", "stop listening",
                                    "go to sleep", "thank you goodbye",
                                    "thanks goodbye", "i'm done", "im done",
                                }

                                words = command.lower().split() if command else []
                                command_clean = command.strip().lower().rstrip(".!?,")

                                is_noise = (
                                    not command
                                    or len(words) < 3
                                    or set(words).issubset(NOISE_WORDS)
                                    or command_clean in HALLUCINATIONS
                                )
                                is_goodbye = any(
                                    phrase in command_clean
                                    for phrase in GOODBYE_PHRASES
                                )

                                if is_goodbye:
                                    print(f"👋 Goodbye detected: '{command}' — stopping listening.")
                                    self.iris.speak("Goodbye! Feel free to ask me anything again anytime.")
                                    if self.iris.websocket and self.iris.websocket_loop:
                                        asyncio.run_coroutine_threadsafe(
                                            self.iris.websocket.send(json.dumps({"listening": False})),
                                            self.iris.websocket_loop
                                        )
                                elif is_noise:
                                    print(f"⚠️ Ignored likely noise: '{command}' — waiting for wake word again.")
                                    if self.iris.websocket and self.iris.websocket_loop:
                                        asyncio.run_coroutine_threadsafe(
                                            self.iris.websocket.send(json.dumps({"listening": False})),
                                            self.iris.websocket_loop
                                        )
                                else:
                                    self.iris.chat(command)
                            finally:
                                self.iris.is_thinking = False
                                print("\n✅ Iris is ready for the next command.")

                        self.loop.run_in_executor(None, respond)

        finally:
            self.iris.websocket = None
            self.clients.discard(websocket)

    async def _lerp_bone(self, bone_name, target_rotation, duration=0.3, steps=15):
        """
        Smoothly interpolates a bone from its current tracked rotation to the target.
        Uses smoothstep easing for a natural feel.
        """
        step_delay = duration / steps
        start_rotation = self.current_pose.get(bone_name, {axis: 0.0 for axis in target_rotation})

        for i in range(1, steps + 1):
            t = i / steps
            t = t * t * (3 - 2 * t)  # smoothstep easing

            msg = {'bone': bone_name, 'rotation': {}}
            for axis, target_val in target_rotation.items():
                start_val = start_rotation.get(axis, 0.0)
                msg['rotation'][axis] = start_val + (target_val - start_val) * t

            self.send_data(msg)
            self.current_pose[bone_name] = dict(msg['rotation'])
            await asyncio.sleep(step_delay)

    async def _random_idle_loop(self):
        """Periodically triggers a random idle animation."""
        while True:
            await asyncio.sleep(random.uniform(5, 10.0))
            if not self.clients or self.speaking:
                continue

            # await self._idle_wave()

            # choice = random.choice(['wave', 'cross_arms'])
            # if choice == 'wave':
            #     await self._idle_wave()
            # else:
            #     await self._idle_cross_arms()

    async def _idle_wave(self):
        """Smooth waving idle animation."""
        self.wave_animating = True
        self.send_data({'expression': 'happy', 'intensity': 1})

        # 1. LIFT: Raise arm into wave-ready position
        await asyncio.gather(
            self._lerp_bone('rightShoulder',  {'x': -1.0},             duration=0.6, steps=20),
            self._lerp_bone('rightUpperArm',  {'z': 1.3},            duration=0.6, steps=20),
            self._lerp_bone('rightLowerArm',  {'z': -1.75, 'x': 1.4}, duration=0.5, steps=15),
            self._lerp_bone('leftLowerArm',   {'z': -1.3, 'y': -2.0},   duration=0.6, steps=20),
            self._lerp_bone('leftUpperArm',   {'z': -1.3},             duration=0.6, steps=20),
        )
        await asyncio.sleep(0.1)

        # 2. WAVE: Smooth back-and-forth motion
        for _ in range(4):
            await self._lerp_bone('rightLowerArm', {'z': -1.75, 'x': 1.3}, duration=0.18, steps=8)
            await self._lerp_bone('rightLowerArm', {'z': -1.75, 'x': 1.5}, duration=0.18, steps=8)

        # 3. RESET: Return to neutral
        self.send_data({'expression': 'happy', 'intensity': 0})
        await asyncio.gather(
            self._lerp_bone('rightShoulder', {'x': 0.0},            duration=0.8, steps=25),
            self._lerp_bone('rightLowerArm', {'z': 0.0, 'x': 0.0}, duration=0.8, steps=25),
            self._lerp_bone('rightUpperArm', {'z': 1.2},           duration=0.8, steps=25),
            self._lerp_bone('leftLowerArm',  {'z': 0.0, 'y': 0.0}, duration=0.5, steps=25),
            self._lerp_bone('leftUpperArm',  {'z': -1.2},            duration=0.5, steps=25),
        )

        self.wave_animating = False

    async def _random_blink_loop(self):
        while True:
            await asyncio.sleep(random.uniform(3.0, 7.0))

            if self.wave_animating:
                await asyncio.sleep(2)

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