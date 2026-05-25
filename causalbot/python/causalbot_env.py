import gymnasium as gym
from gymnasium import spaces
import numpy as np
import websockets
import asyncio
import json
import threading

class CausalBotEnv(gym.Env):
    """
    OpenAI Gymnasium Environment for CausalBot JS Simulation.
    Communicates with the JS app via WebSockets.
    """
    metadata = {"render_modes": ["human"]}

    def __init__(self):
        super().__init__()
        
        # Action space: continuous movement (dx, dz)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        
        # Observation space: 
        # Robot (x, y, z, rot_y) -> 4
        # Glass (x, y, z) -> 3
        # Box (x, y, z) -> 3
        # Ball (x, y, z) -> 3
        # Total = 13
        self.observation_space = spaces.Box(low=-10.0, high=10.0, shape=(13,), dtype=np.float32)
        
        self.state = np.zeros(13, dtype=np.float32)
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

    async def _ws_handler(self, websocket, path):
        self.websocket = websocket
        print("CausalBot JS Client Connected!")
        try:
            async for message in websocket:
                data = json.loads(message)
                
                if data.get("type") == "state":
                    obs = data.get("observation", [])
                    if len(obs) == 13:
                        self.state = np.array(obs, dtype=np.float32)
                        self._state_event.set()
                elif data.get("type") == "reset_done":
                    obs = data.get("observation", [])
                    if len(obs) == 13:
                        self.state = np.array(obs, dtype=np.float32)
                        self._reset_event.set()
                        
        except websockets.exceptions.ConnectionClosed:
            print("CausalBot JS Client Disconnected")
            self.websocket = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        if self.websocket is None:
            print("Waiting for JS client to connect on ws://localhost:8765...")
            # Wait for connection (simple busy wait for demo)
            while self.websocket is None:
                pass
                
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
        
        # Calculate Reward (Example: get close to the ball)
        # state layout: [rx, ry, rz, rot_y, gx, gy, gz, bx, by, bz, ball_x, ball_y, ball_z]
        rx, rz = self.state[0], self.state[2]
        ball_x, ball_z = self.state[10], self.state[12]
        
        dist_to_ball = np.sqrt((rx - ball_x)**2 + (rz - ball_z)**2)
        reward = -0.01  # Step penalty
        
        terminated = False
        truncated = False
        
        if dist_to_ball < 0.5:
            reward += 10.0
            terminated = True
            
        return self.state, reward, terminated, truncated, {}

    def render(self):
        pass # Render is handled by the browser

    def close(self):
        if self.loop.is_running():
            self.loop.call_soon_threadsafe(self.loop.stop)
        self.ws_thread.join(timeout=1)
