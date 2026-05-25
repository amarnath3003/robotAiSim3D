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
        
        # Observation space: 13 elements
        # [target_distance, target_relative_angle, 11_lidar_rays]
        # Dist: [0, 10], Angle: [-pi, pi], Lidar: [0, 5]
        low_obs = np.array([0.0, -np.pi] + [0.0]*11, dtype=np.float32)
        high_obs = np.array([10.0, np.pi] + [5.0]*11, dtype=np.float32)
        self.observation_space = spaces.Box(low=low_obs, high=high_obs, dtype=np.float32)
        
        self.state = np.zeros(13, dtype=np.float32)
        
        # Training Target Goal (x, z)
        self.target_goal = np.array([0.0, 0.0])
        self.max_steps = 1000
        self.current_step = 0
        self.websocket = None
        self.latest_prompt = None
        
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
                    if len(obs) == 14:
                        self._process_js_obs(obs)
                        self._state_event.set()
                elif data.get("type") == "reset_done":
                    obs = data.get("observation", [])
                    if len(obs) == 14:
                        self._process_js_obs(obs)
                        self._reset_event.set()
                elif data.get("type") == "prompt":
                    self.latest_prompt = data.get("text", "")
                        
        except websockets.exceptions.ConnectionClosed:
            print("CausalBot JS Client Disconnected")
            self.websocket = None

    def _process_js_obs(self, obs_list):
        # JS sends: [rx, rz, heading, 11*lidar]
        rx, rz, heading = obs_list[0], obs_list[1], obs_list[2]
        lidar = obs_list[3:]
        
        # Compute relative target
        dx = self.target_goal[0] - rx
        dz = self.target_goal[1] - rz
        
        target_dist = np.sqrt(dx**2 + dz**2)
        # Angle to target in world space
        target_world_angle = np.arctan2(dx, dz)
        
        # Relative angle (wrap to -pi, pi)
        rel_angle = target_world_angle - heading
        while rel_angle > np.pi: rel_angle -= 2*np.pi
        while rel_angle < -np.pi: rel_angle += 2*np.pi
        
        self.state = np.array([target_dist, rel_angle] + lidar, dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        if self.websocket is None:
            print("Waiting for JS client to connect on ws://localhost:8765...")
            while self.websocket is None:
                time.sleep(0.1) # Prevent CPU spinning
                
        self._reset_event.clear()
        self.current_step = 0
        
        # Spawn random goal in room (-2.5 to 2.5)
        self.target_goal = np.array([
            np.random.uniform(-2.5, 2.5), 
            np.random.uniform(-2.5, 2.5)
        ])
        
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
        
        # Reward shaping
        target_dist = self.state[0]
        lidar_min = np.min(self.state[2:])
        
        self.current_step += 1
        
        # 1. Base cost for time passing
        reward = -0.01 
        
        terminated = False
        truncated = False
        
        # 2. Collision Penalty
        if lidar_min < 0.2:
            reward = -10.0
            terminated = True
            
        # 3. Goal Reached
        if target_dist < 0.4:
            reward = 10.0
            terminated = True
            
        # 4. Dense reward for getting closer
        # We need previous distance, but for simplicity we can just reward inverse distance
        # Or just rely on PPO to learn the gradient. Inverse distance is okay:
        if not terminated:
            reward += (0.1 / (target_dist + 0.1))
            
        # 5. Time limit
        if self.current_step >= self.max_steps:
            truncated = True
            
        return self.state, float(reward), terminated, truncated, {}

    def render(self):
        pass # Render is handled by the browser

    def close(self):
        if self.loop.is_running():
            self.loop.call_soon_threadsafe(self.loop.stop)
        self.ws_thread.join(timeout=1)
