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

    def __init__(self, render_mode="human", rocket_urdf_path=r"C:\Users\Lenovo\Desktop\falcon9_project\assets\rocket.urdf", landing_pad_urdf_path=r"C:\Users\Lenovo\Desktop\falcon9_project\assets\landing_pad.urdf"):
        super().__init__()
        self.render_mode = render_mode
        
        # Action space: [main_thrust, rcs_x, rcs_y, rcs_z]
        self.action_space = spaces.Box(
            low=np.array([0.0, -1.0, -1.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32
        )

    #     #corrected
    #     self.action_space = spaces.Box(
    # low=np.array([-1.0] * 15, dtype=np.float32),  # 4 thrusters * (3 directions + 1 magnitude)
    # high=np.array([1.0] * 15, dtype=np.float32),
    # dtype=np.float32)

        
        # Observation space: pos(3) + vel(3) + quat(4) + angvel(3) + fuel(1) = 14 dims
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(14,), dtype=np.float32
        )
        
        # Physics parameters
        self.max_main_thrust = 30000.0  # Newtons
        self.max_rcs_thrust = 500.0    # Newtons
        self.rocket_mass = 1500.0      # kg
        self.initial_fuel = 1000.0      # kg
        self.fuel_consumption_rate = 5  # kg per second at max thrust
        
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
        self.landing_zone_radius = 7.0  # meters
        self.previous_distance = None
        self.previous_altitude = None
        self.landing_attempts = 0
        self.best_landing_distance = float('inf')
        
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
            # p.resetDebugVisualizerCamera(
            #     cameraDistance=60,
            #     cameraYaw=45,
            #     cameraPitch=-30,
            #     cameraTargetPosition=[0, 0, 10]
            # )
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
        start_z = np.random.uniform(59.0, 69.0)
        
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
        self.previous_distance = None
        self.previous_altitude = None
        self.landing_attempts += 1
        
        # Set material properties for more realistic physics
        p.changeDynamics(self.rocket, -1, restitution=0.1, lateralFriction=0.8)
        p.changeDynamics(self.landing_pad, -1, restitution=0.3, lateralFriction=1.0)
        
        observation = self._get_observation()
        info = self._get_info()
        
        return observation,info

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
            "distance_to_target": distance_to_target,
            "altitude": position[2],
            "speed": np.linalg.norm(velocity),
            "fuel_remaining": self.fuel_remaining,
            "euler_angles": euler,
            "step_count": self.step_count
        }

    def _create_engine_particles(self, position, orientation, thrust_intensity):
        """Create particle effects for main engine thrust"""
        
        # Get rocket's orientation matrix
        rotation_matrix = np.array(p.getMatrixFromQuaternion(orientation)).reshape(3, 3)
        
        # Engine exhaust comes out opposite to thrust direction (downward from rocket)
        exhaust_direction = -rotation_matrix[:, 2]  # Opposite to local z-axis
        
        # Engine position (bottom of rocket)
        engine_local_pos = [0, 0, -3.8]  # Bottom of rocket in local coordinates
        engine_world_pos = position + rotation_matrix @ engine_local_pos
        
        # Create multiple exhaust particles based on thrust intensity
        num_particles = int(thrust_intensity * 4)  # More particles for higher thrust
        
        for i in range(num_particles):
            # Random spread for realistic exhaust plume
            spread = 0.1 + thrust_intensity * 0.5  # Larger spread for higher thrust
            random_offset = np.random.uniform(-spread, spread, 3)
            random_offset[2] *= 0.3  # Less vertical spread
            
            # Particle start position (slightly randomized around engine)
            particle_start = engine_world_pos + random_offset
            
            # Particle end position (exhaust plume)
            exhaust_length = 3.0 + thrust_intensity * 2.0  # Longer plume for higher thrust
            particle_end = particle_start + exhaust_direction * exhaust_length + random_offset * 0.5
            
            # Color based on thrust intensity (blue to orange/red)
            if thrust_intensity < 0.3:
                color = [0.3, 0.5, 1.0]  # Blue flame (low thrust)
            elif thrust_intensity < 0.7:
                color = [0.8, 0.6, 0.2]  # Orange flame (medium thrust)
            else:
                color = [1.0, 0.3, 0.1]  # Red flame (high thrust)
            
            # Add some randomness to color
            color = [c + np.random.uniform(-0.1, 0.1) for c in color]
            color = [max(0, min(1, c)) for c in color]  # Clamp to [0,1]
            
            # Draw particle line
            p.addUserDebugLine(
                particle_start,
                particle_end,
                lineColorRGB=color,
                lineWidth=2.0 + thrust_intensity * 2.0,
                lifeTime=0.05  # Short life for flickering effect
            )
    
    def _create_rcs_particles(self, position, orientation, rcs_x, rcs_y, rcs_z):
        """Create particle effects for RCS thrusters"""
        
        rotation_matrix = np.array(p.getMatrixFromQuaternion(orientation)).reshape(3, 3)
        
        # RCS thruster positions around the rocket (simplified - 4 positions)
        rcs_positions = [
            [2.0, 0, 1.25],    # Right side
            [-2.0, 0, 1.25],   # Left side  
            [0, 2.0, 1.25],    # Front
            [0, -2.0, 1.25]    # Back
        ]
        
        # RCS firing intensity
        rcs_intensities = [abs(rcs_x), abs(rcs_x), abs(rcs_y), abs(rcs_y)]
        
        for i, (local_pos, intensity) in enumerate(zip(rcs_positions, rcs_intensities)):
            if intensity > 0.1:  # Only show particles if thruster is firing
                
                # Transform to world coordinates
                world_pos = position + rotation_matrix @ local_pos
                
                # RCS exhaust direction (perpendicular to rocket)
                if i < 2:  # Side thrusters (roll control)
                    exhaust_dir = rotation_matrix[:, 0] * (1 if i == 0 else -1)  # X-axis
                else:  # Front/back thrusters (pitch control)
                    exhaust_dir = rotation_matrix[:, 1] * (1 if i == 2 else -1)  # Y-axis
                
                # Create small RCS exhaust plume
                num_rcs_particles = int(intensity * 5)
                for j in range(num_rcs_particles):
                    particle_start = world_pos + np.random.uniform(-0.2, 0.2, 3)
                    particle_end = particle_start + exhaust_dir * (0.5 + intensity * 1.0)
                    
                    # White/blue RCS exhaust
                    color = [0.8, 0.9, 1.0]
                    
                    p.addUserDebugLine(
                        particle_start,
                        particle_end,
                        lineColorRGB=color,
                        lineWidth=1.0,
                        lifeTime=0.03
                    )
    
    def step(self, action):
        """Execute one time step in the environment"""
        if self.render_mode == "human":
            r_position, _ = p.getBasePositionAndOrientation(self.rocket)
            if r_position[2]<35:
                p.resetDebugVisualizerCamera(
                    cameraDistance=25,
                    cameraYaw=45,
                    cameraPitch=-30,
                    cameraTargetPosition=[0, 0, 10]
                )

            else:
              p.resetDebugVisualizerCamera(
              cameraDistance=15,          
              cameraYaw=50,              
              cameraPitch=-40,           
              cameraTargetPosition=r_position
             )
          

        action = np.clip(action, self.action_space.low, self.action_space.high)
       
        main_thrust = action[0]
        rcs_x, rcs_y, rcs_z = action[1], action[2], action[3]
        
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
            
            # ADD MAIN ENGINE PARTICLE EFFECTS
            if self.render_mode == "human":
                self._create_engine_particles(position, orientation, main_thrust)
            
            # Consume fuel
            fuel_used = main_thrust * self.fuel_consumption_rate * (1.0/60.0)  # Per frame
            self.fuel_remaining = max(0, self.fuel_remaining - fuel_used)
    
        # Apply RCS thrusters for attitude control
        rcs_torque_strength = self.max_rcs_thrust * 0.01  # Convert to torque scale
        torque = [
            rcs_x * rcs_torque_strength,  # Roll
            rcs_y * rcs_torque_strength,  # Pitch
            rcs_z * rcs_torque_strength   # Yaw
        ]
        p.applyExternalTorque(self.rocket, -1, torque, p.LINK_FRAME)
        
        # ADD RCS PARTICLE EFFECTS
        if self.render_mode == "human" and (abs(rcs_x) > 0.1 or abs(rcs_y) > 0.1 or abs(rcs_z) > 0.1):
            position, orientation = p.getBasePositionAndOrientation(self.rocket)
            self._create_rcs_particles(position, orientation, rcs_x, rcs_y, rcs_z)
        
        # Step simulation multiple times for stability
        for _ in range(4):  # 4 substeps per environment step
            p.stepSimulation()
        
        self.step_count += 1
        
        # Get new observation
        observation = self._get_observation()
        terminated, truncated, landed = self._check_termination(observation)
        reward = self._calculate_reward(observation,landed,terminated)
        info = self._get_info()
        
        # Add landing info to info dict
        info["landed_successfully"] = landed
        
        if self.render_mode == "human":
            time.sleep(1.0/60.0)  # Keep real-time rendering
        print(f'reward : {reward}, terminated : {terminated}')
        # FIX: Return only 5 values as per gymnasium standard
        return observation, reward, terminated, truncated, info

    def _calculate_reward(self, obs, landed, terminated):
        """Improved reward function for better PPO performance"""

        # Parse observation
        pos = obs[0:3]
        vel = obs[3:6] 
        quat = obs[6:10]
        ang_vel = obs[10:13]
        fuel_fraction = obs[13]

        # Calculate key metrics
        horizontal_distance = np.sqrt((pos[0] - self.target_x)**2 + (pos[1] - self.target_y)**2)
        altitude = pos[2]
        speed = np.linalg.norm(vel)
        vertical_speed = abs(vel[2])
        horizontal_speed = np.sqrt(vel[0]**2 + vel[1]**2)
        angular_speed = np.linalg.norm(ang_vel)

        # Calculate upright score
        rotation_matrix = np.array(p.getMatrixFromQuaternion(quat)).reshape(3, 3)
        upright_score = rotation_matrix[2, 2]  # 1.0 = perfectly upright

        # Initialize reward
        reward = 0.0

        # 1. TERMINAL REWARDS (large, sparse)
        if landed:
            # Graduated landing bonus based on precision
            landing_bonus = 1000.0
            if horizontal_distance < 1.0:
                landing_bonus += 500.0  # Perfect landing
            elif horizontal_distance < 2.5:
                landing_bonus += 250.0  # Good landing
            elif horizontal_distance < self.landing_zone_radius:
                landing_bonus += 100.0  # Acceptable landing

            # Speed bonus for gentle landing
            if speed < 1.0:
                landing_bonus += 200.0
            elif speed < 2.0:
                landing_bonus += 100.0

            # Upright bonus
            if upright_score > 0.95:
                landing_bonus += 150.0

            # Fuel efficiency bonus
            landing_bonus += fuel_fraction * 100.0

            # Track best performance
            if horizontal_distance < self.best_landing_distance:
                self.best_landing_distance = horizontal_distance
                landing_bonus += 200.0  # Bonus for personal best

            return landing_bonus

        if terminated and not landed:
            # Graduated crash penalty (less harsh than your -100)
            crash_penalty = -200.0

            # Less penalty if close to target when crashed
            if horizontal_distance < self.landing_zone_radius:
                crash_penalty = -200.0
            if horizontal_distance < 2.5:
                crash_penalty = -150.0

            return crash_penalty

        # 2. PROGRESS REWARDS (dense shaping)

        # Distance progress reward (encourage moving toward target)
        if self.previous_distance is not None:
            distance_progress = self.previous_distance - horizontal_distance
            reward += distance_progress * 5.0  # Reward getting closer
        self.previous_distance = horizontal_distance

        # Altitude progress reward (encourage controlled descent)
        if self.previous_altitude is not None and altitude > 0:
            altitude_progress = self.previous_altitude - altitude
            # Only reward descent when above target, penalize when below and moving away
            if altitude > 1.0:
                reward += altitude_progress * 2.0
            elif altitude < 1.0 and altitude_progress < 0:  # Moving up when should be landing
                reward -= 5.0
        self.previous_altitude = altitude

        # 3. STATE-BASED REWARDS (continuous shaping)

        # Proximity reward (exponentially increasing as you get closer)
        max_distance = 50.0  # Maximum expected distance
        proximity_reward = 10.0 * (1.0 - min(horizontal_distance / max_distance, 1.0))**2
        reward += proximity_reward

        # Altitude-dependent rewards
        if altitude < 5.0:
            # Close to landing - emphasize precision and control
            reward += upright_score * 15.0  # Strong upright bonus when landing
            reward -= vertical_speed * 8.0  # Penalize fast descent
            reward -= horizontal_speed * 10.0  # Penalize horizontal drift
            reward -= angular_speed * 15.0  # Penalize spinning near ground

            # Landing zone bonus when close to ground
            if horizontal_distance < self.landing_zone_radius:
                reward += 20.0

        elif altitude < 15.0:
            # Mid-altitude - encourage stable descent
            reward += upright_score * 8.0
            reward -= vertical_speed * 3.0 if vertical_speed > 3.0 else 0  # Only penalize if too fast
            reward -= horizontal_speed * 2.0
            reward -= angular_speed * 5.0

        else:
            # High altitude - encourage general orientation and approach
            reward += upright_score * 3.0
            reward -= angular_speed * 2.0
            # Small penalty for being too high (encourage descent)
            reward -= (altitude - 15.0) * 0.1

        # 4. FUEL EFFICIENCY
        # Reward fuel conservation, but don't penalize use when necessary
        if fuel_fraction > 0.8:
            reward += 2.0  # Bonus for high fuel
        elif fuel_fraction < 0.1:
            reward -= 5.0  # Penalty for very low fuel

        # 5. STABILITY REWARDS
        # Reward low angular velocity (stable flight)
        if angular_speed < 0.1:
            reward += 3.0
        elif angular_speed > 2.0:
            reward -= angular_speed * 3.0

        # 6. BEHAVIORAL SHAPING
        # Small time penalty to encourage efficiency (but not too harsh)
        reward -= 0.05

        # Encourage being in landing zone even at altitude
        if horizontal_distance < self.landing_zone_radius * 2.0:
            reward += 2.0

        # Bonus for maintaining good approach angle
        if altitude > 2.0 and upright_score > 0.8 and horizontal_distance < 10.0:
            reward += 5.0  # Good approach bonus

        return reward








    # def _check_termination(self, obs):
    #     """Check if episode should terminate"""
    #     pos = obs[0:3]
    #     vel = obs[3:6]
    #     quat = obs[6:10]
        
    #     altitude = pos[2]
    #     speed = np.linalg.norm(vel)
    #     horizontal_distance = np.sqrt((pos[0] - self.target_x)**2 + (pos[1] - self.target_y)**2)
        
    #     # Get upright score - Check if rocket exists first
    #     if self.rocket is None:
    #         return False, True  # Truncated if no rocket
        
    #     try:
    #         # More robust quaternion handling
    #         rotation_matrix = np.array(p.getMatrixFromQuaternion(quat)).reshape(3, 3)
    #         upright_score = rotation_matrix[2, 2]  # Correct indexing for z-component
    #     except Exception as e:
    #         print(f"Error calculating upright score: {e}")
    #         upright_score = 0.0
        
    #     terminated = False
    #     truncated = False
        
    #     # Debug prints (you can remove these later)
    #     print(f"Alt: {altitude:.2f}, Speed: {speed:.2f}, H_Dist: {horizontal_distance:.2f}, Upright: {upright_score:.2f}")
        
    #     # Successful landing conditions
    #     if altitude <= 1:  # On or very close to ground
    #         if (speed < 5.0 and upright_score > 0.6 and horizontal_distance < self.landing_zone_radius):
    #             terminated = True
    #             print("🚀 SUCCESSFUL LANDING!")
    #             return terminated, truncated
            
    #         # Crash conditions - if on ground but conditions not met
    #         elif speed > 3.0:  # More lenient speed threshold
    #             print("💥 CRASHED - Too fast!")
    #             terminated = True
    #             return terminated, truncated
            
    #         elif upright_score < 0.3:  # More lenient upright threshold
    #             print("💥 CRASHED - Tipped over!")
    #             terminated = True
    #             return terminated, truncated
                
    #         elif horizontal_distance > self.landing_zone_radius * 2:  # More lenient distance
    #             print("💥 CRASHED - Missed landing zone!")
    #             terminated = True
    #             return terminated, truncated
        
    #     # Out of bounds - More reasonable bounds
    #     if horizontal_distance > 100.0:
    #         print("🚫 OUT OF BOUNDS - Too far horizontally!")
    #         terminated = True
            
    #     elif altitude > 100.0:
    #         print("🚫 OUT OF BOUNDS - Too high!")
    #         terminated = True
            
    #     elif altitude < -2.0:  # Below ground by significant margin
    #         print("🚫 OUT OF BOUNDS - Underground!")
    #         terminated = True
        
    #     # Time limit
    #     if self.step_count >= self.max_steps:
    #         print("⏰ TIME LIMIT REACHED!")
    #         truncated = True
        
    #     # Fuel depletion check
    #     if self.fuel_remaining <= 0 and altitude > 2.0:
    #         print("⛽ OUT OF FUEL!")
    #         terminated = True
        
    #     return terminated, truncated

#more lenient

    # def _check_termination(self, obs):
    #     """Check if episode should terminate"""
    #     pos = obs[0:3]
    #     vel = obs[3:6]
    #     quat = obs[6:10]
        
    #     altitude = pos[2]
    #     speed = np.linalg.norm(vel)
    #     horizontal_distance = np.sqrt((pos[0] - self.target_x)**2 + (pos[1] - self.target_y)**2)
        
    #     # Get upright score - Check if rocket exists first
    #     if self.rocket is None:
    #         return False, True  # Truncated if no rocket
        
    #     try:
    #         # More robust quaternion handling
    #         rotation_matrix = np.array(p.getMatrixFromQuaternion(quat)).reshape(3, 3)
    #         upright_score = rotation_matrix[2, 2]  # Correct indexing for z-component
    #     except Exception as e:
    #         print(f"Error calculating upright score: {e}")
    #         upright_score = 0.0
        
    #     terminated = False
    #     truncated = False
    #     landed=False
        
    #     # Debug prints (you can remove these later)
    #     print(f"Alt: {altitude:.2f}, Speed: {speed:.2f}, H_Dist: {horizontal_distance:.2f}, Upright: {upright_score:.2f}")
        
    #     # MUCH MORE LENIENT LANDING CONDITIONS
    #     if altitude <= 5.0:  # Increased altitude threshold
    #         # Check for successful landing - very lenient conditions
    #         if speed < 5.0 and  upright_score > 0.8 and  horizontal_distance < self.landing_zone_radius :  # Bigger landing zone
                
    #             # Additional check: if very close to ground and reasonable conditions
    #             if altitude <= 5.0 and speed < 5.0:
    #                 terminated = True
    #                 landed=True
    #                 print("🚀 SUCCESSFUL LANDING!")
    #                 return terminated, truncated ,landed
            
    #         # Only crash if conditions are really bad
    #         if altitude <= 0.5:  # Only check crashes when very close to ground
    #             if speed > 12.0:  # Much higher crash speed threshold
    #                 print("💥 CRASHED - Extremely fast impact!")
    #                 terminated = True
    #                 return terminated, truncated,landed
                
    #             elif upright_score < 0.1:  # Only crash if completely upside down
    #                 print("💥 CRASHED - Completely inverted!")
    #                 terminated = True
    #                 return terminated, truncated,landed
        
    #     # Very lenient out of bounds
    #     if horizontal_distance > 20.0:  # Much larger bounds
    #         print("🚫 OUT OF BOUNDS - Too far horizontally!")
    #         terminated = True
            
    #     elif altitude > 75.0:  # Much higher bounds
    #         print("🚫 OUT OF BOUNDS - Too high!")
    #         terminated = True
            
    #     elif altitude < -5.0:  # More tolerance for being underground
    #         print("🚫 OUT OF BOUNDS - Deep underground!")
    #         terminated = True
        
    #     # Time limit
    #     if self.step_count >= self.max_steps:
    #         print("⏰ TIME LIMIT REACHED!")
    #         truncated = True
        
    #     # Only terminate on fuel depletion if really high up
    #     if self.fuel_remaining <= 0 and altitude > 10.0:
    #         print("⛽ OUT OF FUEL!")
    #         terminated = True
        
    #     return terminated, truncated ,landed   

# Less lenient


    def _check_termination(self, obs):
        """
        Check if the episode should terminate with stricter conditions.
        """
        pos = obs[0:3]
        vel = obs[3:6]
        quat = obs[6:10]
        
        altitude = pos[2]
        speed = np.linalg.norm(vel)
        horizontal_distance = np.sqrt((pos[0] - self.target_x)**2 + (pos[1] - self.target_y)**2)
        
        # Calculate upright score from quaternion
        try:
            rotation_matrix = np.array(p.getMatrixFromQuaternion(quat)).reshape(3, 3)
            # The z-component of the rocket's local z-axis (up vector)
            upright_score = rotation_matrix[2, 2] 
        except Exception:
            # Handle cases with invalid quaternions if they occur
            upright_score = 0.0
            
        terminated = False
        truncated = False
        landed = False
        print(f"Alt: {altitude:.2f}, Speed: {speed:.2f}, H_Dist: {horizontal_distance:.2f}, Upright: {upright_score:.2f}")

        # 1. Check for successful landing (requires being very close to the ground)
        if altitude <= 2.5:
            is_on_target = horizontal_distance < self.landing_zone_radius
            is_slow_enough = speed < 0.5  # Stricter speed requirement
            is_upright = upright_score > 0.98 # Stricter upright requirement (less than ~18 deg tilt)

            if is_on_target and is_slow_enough and is_upright:
                terminated = True
                landed = True
                print("🚀 SUCCESSFUL LANDING!")
                return terminated, truncated, landed
            else:
                # 2. If close to the ground but conditions aren't met, it's a crash
                terminated = True
                landed = False
                print(f"💥 CRASHED! [Speed: {speed:.2f}, Upright: {upright_score:.2f}, Dist: {horizontal_distance:.2f}]")
                return terminated, truncated, landed

        # 3. Check for out-of-bounds conditions
        if horizontal_distance > 70.0 or altitude > 70.0 or altitude < -2.0:
            print("🚫 OUT OF BOUNDS!")
            terminated = True
            return terminated, truncated, landed
            
        # 4. Check for time limit
        if self.step_count >= self.max_steps:
            print("⏰ TIME LIMIT REACHED!")
            truncated = True
            return terminated, truncated, landed
            
        # 5. Check for fuel depletion (if not already on the ground)
        if self.fuel_remaining <= 0 and altitude > 1.0:
            print("⛽ OUT OF FUEL!")
            terminated = True
            return terminated, truncated, landed
            
        return terminated, truncated, landed
    
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