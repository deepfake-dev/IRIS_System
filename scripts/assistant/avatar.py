import numpy as np
import sounddevice as sd
import json
import threading
import asyncio
import websockets
import random

from vlm_handler import IrisAssistant

class IrisAvatar:
    wave_animating = False
    speaking = False
    def __init__(self, port=8080):
        self.port = port
        self.clients = set()
        self.current_pose = {}  # tracks current bone rotations for smooth lerping
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
        try:
            # Set neutral pose and sync the tracker
            neutral = {
                'leftUpperArm':  {'z': -1.2},
                'rightUpperArm': {'z': 1.2},
            }
            for bone, rot in neutral.items():
                self.current_pose[bone] = rot
                await websocket.send(json.dumps({'bone': bone, 'rotation': rot}))
            await websocket.wait_closed()
        finally:
            self.clients.remove(websocket)

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

            choice = random.choice(['wave', 'cross_arms'])
            if choice == 'wave':
                await self._idle_wave()
            else:
                await self._idle_cross_arms()

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

    async def _idle_cross_arms(self):
        """Relaxed crossed-arms idle animation."""
        self.wave_animating = True

        # 1. ENTER: Transition into crossed arms pose
        self.send_data({'expression': 'relaxed', 'intensity': 1})
        self.send_data({"expression": "happy", "intensity": 1.0})

        await asyncio.gather(
            self._lerp_bone('rightUpperArm', {'y': 1.35, 'z': 0.25},           duration=0.6, steps=20),
            self._lerp_bone('leftUpperArm',  {'y': -1.25, 'z': -0.4, 'x': 0.0}, duration=0.6, steps=20),
            self._lerp_bone('leftLowerArm',  {'y': -2.0},                        duration=0.6, steps=20),
            self._lerp_bone('rightLowerArm', {'y': 1.8, 'z': -0.15},             duration=0.6, steps=20),
            self._lerp_bone('leftHand',      {'z': -0.25},                       duration=0.6, steps=20),
            self._lerp_bone('rightHand',     {'z': -0.5, 'x': 0.5, 'y': -0.35}, duration=0.6, steps=20),
            self._lerp_bone('leftUpperLeg',  {'x': 0.15},                        duration=0.6, steps=20),
            self._lerp_bone('leftLowerLeg',  {'x': -0.1},                        duration=0.6, steps=20),
            self._lerp_bone('head',          {'z': -0.25},                       duration=0.6, steps=20),
        )

        # 2. HOLD: Linger in the pose for a natural beat
        await asyncio.sleep(random.uniform(3.0, 6.0))

        # 3. RESET: Return to neutral
        self.send_data({'expression': 'relaxed', 'intensity': 0})
        self.send_data({"expression": "happy", "intensity": 0})

        await asyncio.gather(
            self._lerp_bone('rightUpperArm', {'y': 0.0, 'z': 1.2},            duration=0.8, steps=25),
            self._lerp_bone('leftUpperArm',  {'y': 0.0, 'z': -1.2, 'x': 0.0},  duration=0.8, steps=25),
            self._lerp_bone('leftLowerArm',  {'y': 0.0, 'z': 0.0},             duration=0.8, steps=25),
            self._lerp_bone('rightLowerArm', {'y': 0.0, 'z': 0.0},             duration=0.8, steps=25),
            self._lerp_bone('leftHand',      {'z': 0.0},                        duration=0.8, steps=25),
            self._lerp_bone('rightHand',     {'z': 0.0, 'x': 0.0, 'y': 0.0},  duration=0.8, steps=25),
            self._lerp_bone('leftUpperLeg',  {'x': 0.0},                        duration=0.8, steps=25),
            self._lerp_bone('leftLowerLeg',  {'x': 0.0},                        duration=0.8, steps=25),
            self._lerp_bone('head',          {'z': 0.0},                        duration=0.8, steps=25),
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
    
    def animate_listen(self):
        """Triggers the 'listening' pose animation."""
        print("👂 Avatar is now listening...")
        asyncio.run_coroutine_threadsafe(self._listen_pose_sequence(), self.loop)
    
    async def _listen_pose_sequence(self):
        """Internal coroutine to smooth-transition into the listening pose."""
        self.send_data({ 'expression': 'blinkRight', 'intensity': 1.0 })
        self.send_data({ 'mouth': 0.2 })

        await asyncio.gather(
            self._lerp_bone('leftUpperArm',  {'z': -0.75}, duration=0.5),
            self._lerp_bone('leftLowerArm',  {'z': -2.0},  duration=0.5),
            self._lerp_bone('leftHand',      {'z': -4.5},  duration=0.5),
            self._lerp_bone('rightUpperArm', {'z': -0.25}, duration=0.5),
            self._lerp_bone('rightLowerArm', {'z': -2.5},  duration=0.5),
            self._lerp_bone('rightHand',     {'z': 0.75, 'x': -1.0, 'y': -1.0}, duration=0.5),
            self._lerp_bone('chest',         {'z': -0.25}, duration=0.6),
            self._lerp_bone('leftLowerLeg',  {'x': -1.5}, duration=0.6),
            self._lerp_bone('hips',          {'y': 0.75}, duration=0.6)
        )
    
    def animate_unlisten(self):
        """Resets the avatar from the listening pose back to neutral/idle."""
        print("🔈 Avatar is stopping listening...")
        asyncio.run_coroutine_threadsafe(self._unlisten_pose_sequence(), self.loop)

    async def _unlisten_pose_sequence(self):
        """Internal coroutine to smooth-transition back to idle."""
        self.send_data({ 'expression': 'blinkRight', 'intensity': 0.0 })
        self.send_data({ 'mouth': 0.0 })

        bones_to_zero = ['leftHand', 'rightHand', 'chest', 'leftLowerLeg', 'hips', 'rightShoulder']
        
        tasks = [
            self._lerp_bone('rightLowerArm', {'z': 0.0, 'x': 0.0}, duration=0.8, steps=25),
            self._lerp_bone('rightUpperArm', {'z': 1.2},           duration=0.8, steps=25),
            self._lerp_bone('leftLowerArm',  {'z': 0.0, 'y': 0.0}, duration=0.5, steps=25),
            self._lerp_bone('leftUpperArm',  {'z': -1.2},            duration=0.5, steps=25),
        ]

        for bone in bones_to_zero:
            tasks.append(self._lerp_bone(bone, {'x': 0.0, 'y': 0.0, 'z': 0.0}, duration=0.7))

        await asyncio.gather(*tasks)

    def play(self, audio_data, sample_rate=24000):
        self.speaking = True

        if hasattr(audio_data, "cpu"):
            audio_data = audio_data.cpu().numpy()
        elif hasattr(audio_data, "numpy"):
            audio_data = audio_data.numpy()

        data = audio_data.astype(np.float32)

        def callback(outdata, frames, time, status):
            nonlocal data
            if status:
                print(status)
            chunk_size = len(outdata)
            chunk = data[:chunk_size]
            outdata[:len(chunk)] = chunk.reshape(-1, 1)

            if len(chunk) > 0:
                volume = np.linalg.norm(chunk) / np.sqrt(len(chunk))
                openness = np.clip(volume * 8.0, 0.0, 1.0)
                self.send_data({"mouth": float(openness)})

            data = data[chunk_size:]
            if len(data) == 0:
                raise sd.CallbackStop

        with sd.OutputStream(samplerate=sample_rate, channels=1, callback=callback):
            duration_ms = int(len(audio_data) / sample_rate * 1000)
            sd.sleep(duration_ms + 100)

        self.send_data({"mouth": 0.0})
        self.speaking = False

if __name__ == "__main__":
    avatar = IrisAvatar(port=8080)
    bot = IrisAssistant(speaker=avatar)
    bot.run_forever()