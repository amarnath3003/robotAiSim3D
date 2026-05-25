import gymnasium as gym
from gymnasium import spaces
import numpy as np
import websockets
import asyncio
import json
import threading
import time

class CausalBotEnv(gym.Env):
    """
    OpenAI Gymnasium Environment for CausalBot JS Simulation.
    Communicates with the JS app via WebSockets.
    """
    metadata = {"render_modes": ["human"]}

    def __init__(self):
        super().__init__()
        
        # Action space: continuous movement [linear, angular]
        # Linear: [0, 2.5], Angular: [-2.0, 2.0]
        self.action_space = spaces.Box(
            low=np.array([0.0, -2.0]), 
            high=np.array([2.5, 2.0]), 
            dtype=np.float32
        )
        
        # Observation space: 11 lidar rays (distances)
        self.observation_space = spaces.Box(low=0.0, high=10.0, shape=(11,), dtype=np.float32)
        
        self.state = np.zeros(11, dtype=np.float32)
        self.websocket = None
        
        # Used to block gym step until JS sends the next state
        self._state_event = threading.Event()
        self._reset_event = threading.Event()
        
        # Start WebSocket server in a background thread
        self.loop = asyncio.new_event_loop()
        self.ws_thread = threading.Thread(target=self._start_ws_server, daemon=True)
        self.ws_thread.start()

    def _start_ws_server(self):
        asyncio.set_event_loop(self.loop)
        
        async def run_server():
            async with websockets.serve(self._ws_handler, "localhost", 8765):
                await asyncio.Future()  # run forever
                
        self.loop.run_until_complete(run_server())

    async def _ws_handler(self, websocket):
        if self.websocket is not None:
            print("Warning: New JS client connected, closing old connection.")
            try:
                await self.websocket.close()
            except:
                pass
                
        self.websocket = websocket
        print("CausalBot JS Client Connected!")
        try:
            async for message in websocket:
                data = json.loads(message)
                
                if data.get("type") == "state":
                    obs = data.get("observation", [])
                    if len(obs) == 11:
                        self.state = np.array(obs, dtype=np.float32)
                        self._state_event.set()
                elif data.get("type") == "reset_done":
                    obs = data.get("observation", [])
                    if len(obs) == 11:
                        self.state = np.array(obs, dtype=np.float32)
                        self._reset_event.set()
                        
        except websockets.exceptions.ConnectionClosed:
            print("CausalBot JS Client Disconnected")
            self.websocket = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        if self.websocket is None:
            print("Waiting for JS client to connect on ws://localhost:8765...")
            while self.websocket is None:
                time.sleep(0.1) # Prevent CPU spinning
                
        self._reset_event.clear()
        
        # Send reset command to JS
        reset_msg = json.dumps({"type": "reset"})
        asyncio.run_coroutine_threadsafe(self.websocket.send(reset_msg), self.loop)
        
        # Wait for JS to reply with initial state
        self._reset_event.wait()
        
        return self.state, {}

    def step(self, action):
        if self.websocket is None:
            return self.state, 0, True, False, {}
            
        self._state_event.clear()
        
        # Send action to JS
        action_msg = json.dumps({
            "type": "action",
            "action": [float(action[0]), float(action[1])]
        })
        asyncio.run_coroutine_threadsafe(self.websocket.send(action_msg), self.loop)
        
        # Wait for JS to reply with new state
        self._state_event.wait()
        
        # Calculate Reward (Explore without colliding)
        # If any lidar ray is very close, penalize heavily
        min_dist = np.min(self.state)
        
        reward = 0.1 # Living/exploration reward
        terminated = False
        truncated = False
        
        if min_dist < 0.2: # Collision
            reward = -10.0
            terminated = True
        elif min_dist < 0.5: # Danger close
            reward = -1.0
            
        return self.state, reward, terminated, truncated, {}

    def render(self):
        pass # Render is handled by the browser

    def close(self):
        if self.loop.is_running():
            self.loop.call_soon_threadsafe(self.loop.stop)
        self.ws_thread.join(timeout=1)
