import os
import time
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import pybullet as p
import pybullet_data
import tempfile
import textwrap

class Falcon9LandingEnv(gym.Env):
    """
    Gymnasium environment for Falcon 9 rocket landing simulation using PyBullet.
    
    Action Space:
        - main_thrust: [0, 1] - Main engine throttle (0 = off, 1 = max thrust)
        - rcs_x: [-1, 1] - RCS thruster for roll control
        - rcs_y: [-1, 1] - RCS thruster for pitch control  
        - rcs_z: [-1, 1] - RCS thruster for yaw control
        
    Observation Space:
        - position: [x, y, z] - Rocket position in meters
        - velocity: [vx, vy, vz] - Linear velocity in m/s
        - orientation: [qx, qy, qz, qw] - Quaternion orientation
        - angular_velocity: [wx, wy, wz] - Angular velocity in rad/s
        - fuel_remaining: [0, 1] - Remaining fuel fraction
    """
    
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 60,
    }

    def __init__(self, render_mode="human", rocket_urdf_path=r"C:\Users\Lenovo\Desktop\falcon9_project\rocket.urdf", landing_pad_urdf_path=r"C:\Users\Lenovo\Desktop\falcon9_project\landing_pad.urdf"):
        super().__init__()
        self.render_mode = render_mode
        
        # Action space: [main_thrust, rcs_x, rcs_y, rcs_z]
        # self.action_space = spaces.Box(
        #     low=np.array([0.0, -1.0, -1.0, -1.0], dtype=np.float32),
        #     high=np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32),
        #     dtype=np.float32
        # )

        #corrected
        self.action_space = spaces.Box(
    low=np.array([-1.0] * 15, dtype=np.float32),  # 4 thrusters * (3 directions + 1 magnitude)
    high=np.array([1.0] * 15, dtype=np.float32),
    dtype=np.float32)

        
        # Observation space: pos(3) + vel(3) + quat(4) + angvel(3) + fuel(1) = 14 dims
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(14,), dtype=np.float32
        )
        
        # Physics parameters
        self.max_main_thrust = 25000.0  # Newtons
        self.max_rcs_thrust = 5000.0    # Newtons
        self.rocket_mass = 2000.0       # kg
        self.initial_fuel = 1000.0      # kg
        self.fuel_consumption_rate = 0.5  # kg per second at max thrust
        
        # Environment state
        self.client = None
        self.rocket = None
        self.landing_pad = None
        self.ground = None
        self.step_count = 0
        self.max_steps = 1000
        self.fuel_remaining = self.initial_fuel
        
        # Target landing zone
        self.target_x = 0.0
        self.target_y = 0.0
        self.landing_zone_radius = 5.0  # meters
        
        # Set paths to your URDF files
        self._setup_urdf_paths(rocket_urdf_path, landing_pad_urdf_path)

    def _setup_urdf_paths(self, rocket_path=None, pad_path=None):
        """Setup paths to URDF files - use provided paths or search for them"""
        
        # If paths are provided directly, use them
        if rocket_path is not None:
            if os.path.exists(rocket_path):
                self.rocket_urdf_path = os.path.abspath(rocket_path)
                print(f"-> Using rocket URDF: {self.rocket_urdf_path}")
            else:
                raise FileNotFoundError(f"Rocket URDF not found at: {rocket_path}")
        
        if pad_path is not None:
            if os.path.exists(pad_path):
                self.pad_urdf_path = os.path.abspath(pad_path)
                print(f"-> Using landing pad URDF: {self.pad_urdf_path}")
            else:
                raise FileNotFoundError(f"Landing pad URDF not found at: {pad_path}")
        
        # If paths not provided, search for them automatically
        if rocket_path is None or pad_path is None:
            print("🔍 Searching for URDF files...")
            
            # Look for URDF files in common locations
            possible_locations = [
                ".",  # Current directory
                "./assets",
                "../assets", 
                "./urdf",
                "../urdf",
                os.path.join(os.path.dirname(__file__), "assets"),
                os.path.join(os.path.dirname(__file__), "..", "assets"),
                os.path.join(os.path.dirname(__file__), "urdf"),
            ]
            
            if rocket_path is None:
                self.rocket_urdf_path = None
                # Search for rocket.urdf
                for location in possible_locations:
                    rocket_search_path = os.path.join(location, "rocket.urdf")
                    if os.path.exists(rocket_search_path):
                        self.rocket_urdf_path = os.path.abspath(rocket_search_path)
                        break
                        
            if pad_path is None:
                self.pad_urdf_path = None
                # Search for landing_pad.urdf  
                for location in possible_locations:
                    pad_search_path = os.path.join(location, "landing_pad.urdf")
                    if os.path.exists(pad_search_path):
                        self.pad_urdf_path = os.path.abspath(pad_search_path)
                        break
            
            # Check if files were found
            if self.rocket_urdf_path is None:
                raise FileNotFoundError(
                    "Could not find 'rocket.urdf'. Please either:\n" +
                    "1. Provide rocket_urdf_path parameter, or\n" +
                    "2. Place rocket.urdf in one of these locations:\n" +
                    "\n".join(f"   - {loc}" for loc in possible_locations)
                )
                
            if self.pad_urdf_path is None:
                raise FileNotFoundError(
                    "Could not find 'landing_pad.urdf'. Please either:\n" +
                    "1. Provide landing_pad_urdf_path parameter, or\n" +
                    "2. Place landing_pad.urdf in one of these locations:\n" +
                    "\n".join(f"   - {loc}" for loc in possible_locations)
                )
                
            print(f"📁 Found rocket URDF: {self.rocket_urdf_path}")
            print(f"📁 Found landing pad URDF: {self.pad_urdf_path}")

    def _connect_physics(self):
        """Initialize PyBullet physics simulation"""
        if self.client is not None:
            return
        if self.render_mode == "human":
            self.client = p.connect(p.GUI)
            # Set up nice camera view
            p.resetDebugVisualizerCamera(
                cameraDistance=25,
                cameraYaw=45,
                cameraPitch=-30,
                cameraTargetPosition=[0, 0, 10]
            )
        else:
            self.client = p.connect(p.DIRECT)
            
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        p.setPhysicsEngineParameter(
            fixedTimeStep=1.0/240.0,  # High frequency for stability
            numSolverIterations=20
        )
        p.setRealTimeSimulation(0)

    def reset(self, seed=None, options=None):
        """Reset the environment to initial state"""
        super().reset(seed=seed)
        
        self._connect_physics()
        p.resetSimulation()
        p.setGravity(0, 0, -9.81)
        
        # Load ground plane
        self.ground = p.loadURDF("plane.urdf")
        
        # Load landing pad
        self.landing_pad = p.loadURDF(
            self.pad_urdf_path,
            basePosition=[self.target_x, self.target_y, 0.0],  # Place at ground level
            useFixedBase=True
        )
        
        # Load rocket with random initial conditions
        if seed is not None:
            np.random.seed(seed)
            
        # Random starting position (higher altitude, some horizontal offset)
        start_x = np.random.uniform(-5.0, 5.0)
        start_y = np.random.uniform(-5.0, 5.0)
        start_z = np.random.uniform(15.0, 25.0)
        
        # Random starting orientation (small perturbations)
        roll = np.random.uniform(-0.2, 0.2)
        pitch = np.random.uniform(-0.2, 0.2) 
        yaw = np.random.uniform(-np.pi, np.pi)
        
        start_orientation = p.getQuaternionFromEuler([roll, pitch, yaw])
        
        self.rocket = p.loadURDF(
            self.rocket_urdf_path,
            basePosition=[start_x, start_y, start_z],
            baseOrientation=start_orientation,
            useFixedBase=False
        )
        
        # Add some initial velocity for realism
        initial_vel = [
            np.random.uniform(-2.0, 2.0),  # vx
            np.random.uniform(-2.0, 2.0),  # vy
            np.random.uniform(-5.0, -1.0)  # vz (falling)
        ]
        p.resetBaseVelocity(self.rocket, linearVelocity=initial_vel)
        
        # Reset environment state
        self.step_count = 0
        self.fuel_remaining = self.initial_fuel
        
        # Set material properties for more realistic physics
        p.changeDynamics(self.rocket, -1, restitution=0.1, lateralFriction=0.8)
        p.changeDynamics(self.landing_pad, -1, restitution=0.3, lateralFriction=1.0)
        
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, info

    def _get_observation(self):
        """Get current observation of the rocket state"""
        position, orientation = p.getBasePositionAndOrientation(self.rocket)
        velocity, angular_velocity = p.getBaseVelocity(self.rocket)
        # print(position,orientation)
        
        # Normalize fuel remaining
        fuel_fraction = self.fuel_remaining / self.initial_fuel
        
        observation = np.array([
            position[0], position[1], position[2],           # position (3)
            velocity[0], velocity[1], velocity[2],           # velocity (3)
            orientation[0], orientation[1], orientation[2], orientation[3],  # quaternion (4)
            angular_velocity[0], angular_velocity[1], angular_velocity[2],   # angular velocity (3)
            fuel_fraction                                    # fuel remaining (1)
        ], dtype=np.float32)
        
        return observation

    def _get_info(self):
        """Get additional info dict"""
        position, orientation = p.getBasePositionAndOrientation(self.rocket)
        velocity, _ = p.getBaseVelocity(self.rocket)
        
        distance_to_target = np.sqrt(
            (position[0] - self.target_x)**2 + 
            (position[1] - self.target_y)**2
        )
        
        # Convert quaternion to euler for easier interpretation
        euler = p.getEulerFromQuaternion(orientation)
        
        return {
            "distance_to_target": {distance_to_target},
            "altitude": position[2],
            "speed": np.linalg.norm(velocity),
            "fuel_remaining": self.fuel_remaining,
            "euler_angles": euler,
            "step_count": self.step_count
        }

    def step(self, action):
        """Execute one time step in the environment"""
        action = np.clip(action, self.action_space.low, self.action_space.high)
       
        main_thrust = action[0]
        rcs_x, rcs_y, rcs_z = action[1], action[2], action[3]
        # print(f'main:{main_thrust},rcs_x:{rcs_x},y:{rcs_y},z:{rcs_z}')
        # Check if we have fuel
        if self.fuel_remaining <= 0:
            main_thrust = 0.0
            rcs_x = rcs_y = rcs_z = 0.0
        
        # Apply main engine thrust
        if main_thrust > 0.01:  # Threshold to avoid tiny thrusts
            thrust_force = main_thrust * self.max_main_thrust
            position, orientation = p.getBasePositionAndOrientation(self.rocket)
            
            # Get rocket's up direction (local z-axis in world coordinates)
            rotation_matrix = np.array(p.getMatrixFromQuaternion(orientation)).reshape(3, 3)
            thrust_direction = rotation_matrix[:, 2]  # Local z-axis
            
            thrust_vector = thrust_direction * thrust_force
            
            # Apply thrust at rocket's center of mass with small offset for realism
            thrust_point = [0, 0, -4.0]  # Bottom of rocket in local coordinates
            p.applyExternalForce(
                self.rocket, -1, thrust_vector, thrust_point, p.LINK_FRAME
            )
            
            # Consume fuel
            fuel_used = main_thrust * self.fuel_consumption_rate * (1.0/60.0)  # Per frame
            self.fuel_remaining = max(0, self.fuel_remaining - fuel_used)

        # Apply RCS thrusters for attitude control
        rcs_torque_strength = self.max_rcs_thrust * 0.001  # Convert to torque scale
        torque = [
            rcs_x * rcs_torque_strength,  # Roll
            rcs_y * rcs_torque_strength,  # Pitch
            rcs_z * rcs_torque_strength   # Yaw
        ]
        p.applyExternalTorque(self.rocket, -1, torque, p.LINK_FRAME)
        
        # Step simulation multiple times for stability
        for _ in range(4):  # 4 substeps per environment step
            p.stepSimulation()
        
        self.step_count += 1
        
        # Get new observation
        observation = self._get_observation()
        reward = self._calculate_reward(observation)
        terminated, truncated = self._check_termination(observation)
        info = self._get_info()
        
        if self.render_mode == "human":
            time.sleep(1.0/60.0)  # Keep real-time rendering
        
        return observation, reward, terminated, truncated, info

    def _calculate_reward(self, obs):
        """Calculate reward based on current state"""
        pos = obs[0:3]
        vel = obs[3:6] 
        quat = obs[6:10]
        ang_vel = obs[10:13]
        fuel_frac = obs[13]
        
        # Distance from target landing zone
        horizontal_distance = np.sqrt((pos[0] - self.target_x)**2 + (pos[1] - self.target_y)**2)
        altitude = pos[2]
        speed = np.linalg.norm(vel)
        angular_speed = np.linalg.norm(ang_vel)
        
        # Compute upright orientation bonus
        # Convert quaternion to rotation matrix and get z-axis component
        rotation_matrix = np.array(p.getMatrixFromQuaternion(quat)).reshape(3, 3)
        up_vector = rotation_matrix[:, 2]  # Local z-axis in world coordinates
        upright_score = up_vector[2]  # How much the rocket points up (1.0 = perfectly upright)
        
        # Reward components
        reward = 0.0
        
        # Encourage staying close to target horizontally
        reward -= horizontal_distance * 2.0
        
        # Encourage being upright
        reward += upright_score * 5.0
        
        # Penalize high speeds (encourage gentle landing)
        reward -= speed * 0.5
        reward -= angular_speed * 2.0
        
        # Small fuel efficiency bonus
        reward += fuel_frac * 0.1
        
        # Height-dependent rewards (encourage controlled descent)
        if altitude < 2.0:
            # Close to landing - require very controlled approach
            reward -= speed * 5.0  # Heavy penalty for fast landing
            reward += upright_score * 10.0  # Heavy bonus for being upright
            
            if horizontal_distance < self.landing_zone_radius:
                reward += 20.0  # In landing zone
                
        return reward

    # def _check_termination(self, obs):
    #     """Check if episode should terminate"""
    #     pos = obs[0:3]
    #     vel = obs[3:6]
    #     quat = obs[6:10]
    #     print(f'pos: {pos}')
    #     print(f'vel :{vel}')
    #     print(f'quat :{quat }')
        
    #     altitude = pos[2]
    #     speed = np.linalg.norm(vel)
    #     horizontal_distance = np.sqrt((pos[0] - self.target_x)**2 + (pos[1] - self.target_y)**2)
    #     print(f'speed:{speed}')
    #     # Get upright score
    #     rotation_matrix = np.array(p.getMatrixFromQuaternion(quat)).reshape(3, 3)
    #     upright_score = rotation_matrix[2,2]
        
    #     terminated = False
    #     truncated = False
        
    #     # Successful landing conditions
    #     if (altitude < 0.5 and speed < 1.0 and upright_score > 0.8 and 
    #         horizontal_distance < self.landing_zone_radius):
    #         terminated = True
            
    #     # Crash conditions
    #     elif altitude < 0.2 and (speed > 1.0 ):
    #         terminated = True
            
    #     # Out of bounds
    #     elif horizontal_distance > 50.0 or altitude > 50.0:
    #         terminated = True
            
    #     # Time limit
    #     elif self.step_count >= self.max_steps:
    #         truncated = True
            
    #     return terminated, truncated

    def _check_termination(self, obs):
        """Check if episode should terminate"""
        pos = obs[0:3]
        vel = obs[3:6]
        quat = obs[6:10]
        
        altitude = pos[2]
        speed = np.linalg.norm(vel)
        horizontal_distance = np.sqrt((pos[0] - self.target_x)**2 + (pos[1] - self.target_y)**2)
        
        # Get upright score - Check if rocket exists first
        if self.rocket is None:
            return False, True  # Truncated if no rocket
        
        try:
            # More robust quaternion handling
            rotation_matrix = np.array(p.getMatrixFromQuaternion(quat)).reshape(3, 3)
            upright_score = rotation_matrix[2, 2]  # Correct indexing for z-component
        except Exception as e:
            print(f"Error calculating upright score: {e}")
            upright_score = 0.0
        
        terminated = False
        truncated = False
        
        # Debug prints (you can remove these later)
        print(f"Alt: {altitude:.2f}, Speed: {speed:.2f}, H_Dist: {horizontal_distance:.2f}, Upright: {upright_score:.2f}")
        
        # Successful landing conditions
        if altitude <= 0.5:  # On or very close to ground
            if (speed < 5.0 and upright_score > 0.6 and horizontal_distance < self.landing_zone_radius):
                terminated = True
                print("🚀 SUCCESSFUL LANDING!")
                return terminated, truncated
            
            # Crash conditions - if on ground but conditions not met
            elif speed > 3.0:  # More lenient speed threshold
                print("💥 CRASHED - Too fast!")
                terminated = True
                return terminated, truncated
            
            elif upright_score < 0.3:  # More lenient upright threshold
                print("💥 CRASHED - Tipped over!")
                terminated = True
                return terminated, truncated
                
            elif horizontal_distance > self.landing_zone_radius * 2:  # More lenient distance
                print("💥 CRASHED - Missed landing zone!")
                terminated = True
                return terminated, truncated
        
        # Out of bounds - More reasonable bounds
        if horizontal_distance > 100.0:
            print("🚫 OUT OF BOUNDS - Too far horizontally!")
            terminated = True
            
        elif altitude > 100.0:
            print("🚫 OUT OF BOUNDS - Too high!")
            terminated = True
            
        elif altitude < -2.0:  # Below ground by significant margin
            print("🚫 OUT OF BOUNDS - Underground!")
            terminated = True
        
        # Time limit
        if self.step_count >= self.max_steps:
            print("⏰ TIME LIMIT REACHED!")
            truncated = True
        
        # Fuel depletion check
        if self.fuel_remaining <= 0 and altitude > 2.0:
            print("⛽ OUT OF FUEL!")
            terminated = True
        
        return terminated, truncated

    def render(self):
        """Render the environment"""
        if self.render_mode == "human":
            # Add some debug information
            if hasattr(self, 'rocket') and self.rocket is not None:
                pos, _ = p.getBasePositionAndOrientation(self.rocket)
                info = self._get_info()
                
                # Display information in GUI
                debug_text = f"Altitude: {pos[2]:.1f}m | Distance: {info['distance_to_target']:.1f}m | Fuel: {self.fuel_remaining:.0f}kg"
                p.addUserDebugText(debug_text, [pos[0], pos[1], pos[2] + 8], textSize=1.5, lifeTime=0.1)
                
                # Draw target zone
                p.addUserDebugLine(
                    [self.target_x - self.landing_zone_radius, self.target_y, 0.3],
                    [self.target_x + self.landing_zone_radius, self.target_y, 0.3],
                    [1, 0, 0], lineWidth=3, lifeTime=0.1
                )
                p.addUserDebugLine(
                    [self.target_x, self.target_y - self.landing_zone_radius, 0.3],
                    [self.target_x, self.target_y + self.landing_zone_radius, 0.3],
                    [1, 0, 0], lineWidth=3, lifeTime=0.1
                )

    def close(self):
        """Clean up resources"""
        if self.client is not None:
            p.disconnect(self.client)
            self.client = None


# Example usage and testing
# if __name__ == "__main__":
#     # Example 1: Use automatic detection (your files in current folder)
#     print("Example 1: Automatic URDF detection")
#     try:
#         env = Falcon9LandingEnv(render_mode="human")
#         print("✅ Environment created with auto-detected files")
#         obs, info = env.reset()
#         env.close()
#     except Exception as e:
#         print(f"❌ Auto-detection failed: {e}")
    
#     print("\n" + "="*50)
    
#     # Example 2: Direct path specification
#     print("Example 2: Direct path specification")
    
#     # PASTE YOUR PATHS HERE:
#     rocket_path = r"C:\Users\Lenovo\Desktop\falcon9_project\rocket.urdf"
#     pad_path = r"C:\Users\Lenovo\Desktop\falcon9_project\landing_pad.urdf"
    
#     try:
#         env = Falcon9LandingEnv(
#             render_mode="human",
#             rocket_urdf_path=rocket_path,
#             landing_pad_urdf_path=pad_path
#         )
#         print("✅ Environment created with direct paths")
        
#         # Test the environment
#         obs, info = env.reset()
#         print(f"Initial observation shape: {obs.shape}")
#         print(f"Initial altitude: {obs[2]:.1f}m")
        
#         # Run a few steps to test
#         for step in range(100):
#             # Simple hover test
#             action = np.array([0.45, 0.0, 0.0, 0.0])  # Light thrust, no RCS
#             obs, reward, terminated, truncated, info = env.step(action)
            
#             if step % 20 == 0:
#                 print(f"Step {step}: Alt={obs[2]:.1f}m, Reward={reward:.1f}")
            
#             if terminated or truncated:
#                 print(f"Episode ended at step {step}")
#                 break
        
#         env.close()
#         print("✅ Environment test completed successfully!")
        
#     except Exception as e:
#         print(f"❌ Direct path test failed: {e}")
#         import traceback
#         traceback.print_exc()