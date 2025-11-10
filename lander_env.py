import os
import time
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import pybullet as p
import pybullet_data

class Falcon9LandingEnv(gym.Env):
    """
    Gymnasium environment for Falcon 9 rocket landing simulation using PyBullet.
    
    Action Space:
        - main_thrust: [0, 1] - Main engine throttle (0 = off, 1 = max thrust)
        - thruster_north: [0, 1] - North thruster (front of rocket)
        - thruster_east: [0, 1] - East thruster (right side of rocket)
        - thruster_south: [0, 1] - South thruster (back of rocket)  
        - thruster_west: [0, 1] - West thruster (left side of rocket)
        
    Observation Space:
        - position: [x, y, z] - Rocket position in meters
        - velocity: [vx, vy, vz] - Linear velocity in m/s
        - orientation: [qx, qy, qz, qw] - Quaternion orientation
        - angular_velocity: [wx, wy, wz] - Angular velocity in rad/s
        - fuel_remaining: [0, 1] - Remaining fuel fraction
    """
    
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 50,
    }

    def __init__(self, render_mode="human",difficulty='hard', rocket_urdf_path=r"C:\Users\Lenovo\Desktop\falcon9_project\assets\rocket.urdf", landing_pad_urdf_path=r"C:\Users\Lenovo\Desktop\falcon9_project\assets\landing_pad.urdf"):
        super().__init__()
        self.render_mode = render_mode
        self.difficulty=difficulty
        # Action space: [main_thrust, thruster_north, thruster_east, thruster_south, thruster_west]
        self.action_space = spaces.Box(
            low=np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32
        )
        
        # Observation space: pos(3) + vel(3) + quat(4) + angvel(3) + fuel(1) = 14 dims
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(14,), dtype=np.float32
        )
        
        # Physics parameters
        self.max_main_thrust = 130000.0  # Newtons
        self.max_thruster_force = 5000.0  # Newtons per individual thruster
        self.rocket_mass = 1500.0      # kg
        self.initial_fuel = 1000.0      # kg
        self.fuel_consumption_rate = 7.5 # kg per second at max thrust
        
        # Environment state
        self.client = None
        self.rocket = None
        self.landing_pad = None
        self.ground = None
        self.step_count = 0
        self.max_steps = 1200
        self.fuel_remaining = self.initial_fuel
        
        # Target landing zone
        self.target_x = 0.0
        self.target_y = 0.0
        self.landing_zone_radius = 7.0  # meters
        self.previous_distance = None
        self.previous_altitude = None
        self.landing_attempts = 0
        self.best_landing_distance = float('inf')
        
        # Thruster positions (near top of rocket in local coordinates)
        self.thruster_positions = {
            'north': [0, 2.5, 2.0],   # Front (positive Y)
            'east':  [2.5, 0, 2.0],   # Right (positive X)
            'south': [0, -2.5, 2.0],  # Back (negative Y)
            'west':  [-2.5, 0, 2.0]   # Left (negative X)
        }
        
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
            print(" Searching for URDF files...")
            
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
        if self.difficulty == "easy":
            start_x = np.random.uniform(-1.0, 1.0)
            start_y = np.random.uniform(-1.0, 1.0)
            start_z = np.random.uniform(15.0, 25.0)  # LOW altitude
            self.landing_zone_radius = 10.0
        elif self.difficulty == "medium":
            start_x = np.random.uniform(-2.0, 2.0)
            start_y = np.random.uniform(-2.0, 2.0)
            start_z = np.random.uniform(30.0, 40.0)  # MEDIUM altitude
            self.landing_zone_radius = 7.0
        else:  # hard
            start_x = np.random.uniform(-3.0, 3.0)
            start_y = np.random.uniform(-3.0, 3.0)
            start_z = np.random.uniform(50.0, 60.0)  # HIGH altitude
            self.landing_zone_radius = 7.0      
        
        # Random starting orientation (small perturbations)
        roll = np.random.uniform(-0.0, 0.0)
        pitch = np.random.uniform(-0.0, 0.0) 
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
            np.random.uniform(-0.0, 0.0),  # vx
            np.random.uniform(-0.0, 0.0),  # vy
            np.random.uniform(-1.0, -0.5)  # vz (falling)
        ]
        p.resetBaseVelocity(self.rocket, linearVelocity=initial_vel)
        
        # Reset environment state
        self.step_count = 0
        self.fuel_remaining = self.initial_fuel
        self.previous_distance = None
        self.previous_altitude = None
        self.landing_attempts += 1
        
        # Set material properties for more realistic physics
        p.changeDynamics(self.rocket, -1, linearDamping=0.02)
        p.changeDynamics(self.landing_pad, -1, restitution=0.3, lateralFriction=1.0)
        
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, info

    def _get_observation(self):
        """Get current observation of the rocket state"""
        position, orientation = p.getBasePositionAndOrientation(self.rocket)
        velocity, angular_velocity = p.getBaseVelocity(self.rocket)
        
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
    
    def _create_thruster_particles(self, position, orientation, thruster_forces):
        """Create particle effects for individual thrusters"""
        
        rotation_matrix = np.array(p.getMatrixFromQuaternion(orientation)).reshape(3, 3)
        
        # Thruster names and their corresponding forces
        thruster_names = ['north', 'east', 'south', 'west']
        
        for i, (thruster_name, force) in enumerate(zip(thruster_names, thruster_forces)):
            if force > 0.05:  # Only show particles if thruster is firing significantly
                
                # Get thruster position in world coordinates
                local_pos = self.thruster_positions[thruster_name]
                world_pos = position + rotation_matrix @ local_pos
                
                # Determine exhaust direction based on thruster position
                # Thrusters push outward from rocket center
                if thruster_name == 'north':   # Front thruster pushes forward
                    exhaust_dir = rotation_matrix[:, 1]  # +Y direction
                elif thruster_name == 'south': # Back thruster pushes backward
                    exhaust_dir = -rotation_matrix[:, 1]  # -Y direction
                elif thruster_name == 'east':  # Right thruster pushes right
                    exhaust_dir = rotation_matrix[:, 0]  # +X direction
                elif thruster_name == 'west':  # Left thruster pushes left
                    exhaust_dir = -rotation_matrix[:, 0]  # -X direction
                
                # Create thruster exhaust particles
                num_particles = int(force * 3)  # Number based on thrust intensity
                
                for j in range(num_particles):
                    # Small random offset around thruster position
                    particle_start = world_pos + np.random.uniform(-0.1, 0.1, 3)
                    
                    # Exhaust length based on force
                    exhaust_length = 0.5 + force * 1.5
                    particle_end = particle_start + exhaust_dir * exhaust_length
                    
                    # Thruster color (white/blue for cold gas thrusters)
                    color = [0.7, 0.8, 1.0]  # Light blue
                    
                    p.addUserDebugLine(
                        particle_start,
                        particle_end,
                        lineColorRGB=color,
                        lineWidth=1.5,
                        lifeTime=0.04  # Short life for realistic effect
                    )

    def step(self, action):
        """Execute one time step in the environment"""
        if self.render_mode == "human":
            r_position, _ = p.getBasePositionAndOrientation(self.rocket)
            if r_position[2] < 35:
                p.resetDebugVisualizerCamera(
                    cameraDistance=20,
                    cameraYaw=0,
                    cameraPitch=-5,
                    cameraTargetPosition=[0, 0, 5]
                )
            else:
                p.resetDebugVisualizerCamera(
                    cameraDistance=15,          
                    cameraYaw=10,              
                    cameraPitch=-40,           
                    cameraTargetPosition=r_position
                )

        action = np.clip(action, self.action_space.low, self.action_space.high)
       
        main_thrust = action[0]
        thruster_north = action[1]
        thruster_east = action[2]
        thruster_south = action[3]
        thruster_west = action[4]
        
        # Check if we have fuel
        if self.fuel_remaining <= 0:
            main_thrust = 0.0
            thruster_north = thruster_east = thruster_south = thruster_west = 0.0
        
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
            
            # Add main engine particle effects
            if self.render_mode == "human":
                self._create_engine_particles(position, orientation, main_thrust)
            
            # Consume fuel
            fuel_used = main_thrust * self.fuel_consumption_rate * (1.0/60.0)  # Per frame
            self.fuel_remaining = max(0, self.fuel_remaining - fuel_used)
    
        # Apply individual thrusters
        position, orientation = p.getBasePositionAndOrientation(self.rocket)
        rotation_matrix = np.array(p.getMatrixFromQuaternion(orientation)).reshape(3, 3)
        
        thruster_forces = [thruster_north, thruster_east, thruster_south, thruster_west]
        thruster_names = ['north', 'east', 'south', 'west']
        
        for i, (thruster_name, thrust_level) in enumerate(zip(thruster_names, thruster_forces)):
            if thrust_level > 0.01:
                
                # Calculate force magnitude
                force_magnitude = thrust_level * self.max_thruster_force
                
                # Get thruster position in world coordinates
                local_pos = self.thruster_positions[thruster_name]
                thruster_world_pos = position + rotation_matrix @ local_pos
                
                # Determine force direction (opposite to exhaust direction)
                if thruster_name == 'north':   # North thruster pushes rocket south
                    force_direction = -rotation_matrix[:, 1]  # -Y direction
                elif thruster_name == 'south': # South thruster pushes rocket north
                    force_direction = rotation_matrix[:, 1]   # +Y direction
                elif thruster_name == 'east':  # East thruster pushes rocket west
                    force_direction = -rotation_matrix[:, 0]  # -X direction
                elif thruster_name == 'west':  # West thruster pushes rocket east
                    force_direction = rotation_matrix[:, 0]   # +X direction
                
                # Apply force at thruster location
                force_vector = force_direction * force_magnitude
                p.applyExternalForce(
                    self.rocket, -1, force_vector, local_pos, p.LINK_FRAME
                )
                
                # Small fuel consumption for thrusters
                fuel_used = thrust_level * 0.2 * (1.0/60.0)  # Less fuel than main engine
                self.fuel_remaining = max(0, self.fuel_remaining - fuel_used)
        
        # Add thruster particle effects
        if self.render_mode == "human" and any(f > 0.05 for f in thruster_forces):
            self._create_thruster_particles(position, orientation, thruster_forces)
        
        # Step simulation multiple times for stability
        for _ in range(4):  # 4 substeps per environment step
            p.stepSimulation()
        
        self.step_count += 1
        
        # Get new observation
        observation = self._get_observation()
        terminated, truncated, landed = self._check_termination(observation)
        reward = self._calculate_reward(observation, landed, terminated)
        info = self._get_info()
        
        # Add landing info to info dict
        info["landed_successfully"] = landed
        
        if self.render_mode == "human":
            time.sleep(1.0/60.0)  # Keep real-time rendering
            
        print(f'reward : {reward}, terminated : {terminated}')
        
        return observation, reward, terminated, truncated, info

    # def _calculate_reward(self, obs, landed, terminated):
    #     """
    #     Improved reward function with better scaling and clearer objectives.
    #     Key principles:
    #     1. Consistent reward scaling (-1 to +1 per step, large terminal rewards)
    #     2. Clear hierarchical objectives (altitude > orientation > position)
    #     3. Smooth reward gradients for better learning
    #     """
        
    #     # Parse observation
    #     pos = obs[0:3]
    #     vel = obs[3:6]
    #     quat = obs[6:10]
    #     ang_vel = obs[10:13]
    #     fuel_fraction = obs[13]
        
    #     # Calculate key metrics
    #     x, y, z = pos
    #     vx, vy, vz = vel
    #     altitude = z
        
    #     horizontal_distance = np.sqrt((x - self.target_x)**2 + (y - self.target_y)**2)
    #     speed = np.linalg.norm(vel)
    #     vertical_speed = abs(vz)
    #     horizontal_speed = np.sqrt(vx**2 + vy**2)
    #     angular_speed = np.linalg.norm(ang_vel)
        
    #     # Calculate upright score (how vertical the rocket is)
    #     rotation_matrix = np.array(p.getMatrixFromQuaternion(quat)).reshape(3, 3)
    #     upright_score = rotation_matrix[2, 2]  # 1.0 = perfectly upright, 0.0 = horizontal
    #     tilt_angle = np.arccos(np.clip(upright_score, -1, 1))  # radians
        
    #     reward = 0.0
        
    #     # ============================================================================
    #     # TERMINAL REWARDS (Large, sparse rewards for episode outcomes)
    #     # ============================================================================
        
    #     if landed:
    #         # Base landing reward
    #         base_reward = 2000.0
            
    #         # Distance precision bonus (exponential falloff)
    #         distance_bonus = 1000.0 * np.exp(-horizontal_distance / 1.5)
            
    #         # Speed bonus (exponential - reward gentle landing)
    #         speed_bonus = 500.0 * np.exp(-speed / 2.0)
            
    #         # Orientation bonus
    #         orientation_bonus = 300.0 * upright_score
            
    #         # Fuel efficiency bonus
    #         fuel_bonus = 500.0 * fuel_fraction
            
    #         total_landing_reward = (base_reward + distance_bonus + 
    #                                speed_bonus + orientation_bonus + fuel_bonus)
            
    #         # Extra bonus for exceptional landing
    #         if horizontal_distance < 1.0 and speed < 1.0 and upright_score > 0.98:
    #             total_landing_reward += 500.0  # Perfect landing bonus
            
    #         print(f"🚀 LANDED! Reward: {total_landing_reward:.0f} "
    #               f"(dist: {horizontal_distance:.2f}m, speed: {speed:.2f}m/s)")
            
    #         return total_landing_reward
        
    #     if terminated and not landed:
    #         # Crash penalty (less harsh if you were close)
    #         base_penalty = -800.0
            
    #         # Reduce penalty if close to target (you were trying!)
    #         if horizontal_distance < self.landing_zone_radius:
    #             base_penalty *= 0.6
    #         if horizontal_distance < 3.0:
    #             base_penalty *= 0.7
                
    #         # Reduce penalty if mostly upright (good orientation control)
    #         if upright_score > 0.8:
    #             base_penalty *= 0.8
                
    #         print(f"💥 CRASHED! Penalty: {base_penalty:.0f}")
    #         return base_penalty
        
    #     # ============================================================================
    #     # STEP REWARDS (Dense shaping rewards, normalized to roughly -1 to +1 per step)
    #     # ============================================================================
        
    #     time_pressure = 0.05 + (self.step_count / self.max_steps) * 0.2
    #     reward -= time_pressure

    #     # 1. ALTITUDE REWARD (Primary objective: controlled descent)
    #     # -------------------------------------------------------------------------
    #     target_altitude = 0.0  # We want to reach ground
    #     altitude_error = altitude - target_altitude
        
        
    #     # Reward being at correct altitude for current phase
    #     if altitude > 20.0:
    #         altitude_reward = -0.1
    #         if vz < -2.0:
    #             altitude_reward += 0.15
    #     elif altitude > 10.0:
    #         altitude_reward = 0.2 - altitude * 0.01
    #         if vz < -2.0:
    #             altitude_reward += 0.2
    #     elif altitude > 3.0:
    #         altitude_reward = 0.5 - altitude * 0.05
    #         if vz < -1.0:
    #             altitude_reward += 0.4
    #         if abs(vz) < 0.5:
    #             altitude_reward -= 0.5
    #     else:
    #         altitude_reward = 0.8
    #         if abs(vz) < 0.2:
    #             altitude_reward -= 1.0
        
    #     reward += altitude_reward
        
    #     # 2. POSITION REWARD (Secondary objective: stay above landing zone)
    #     # -------------------------------------------------------------------------
    #     # Exponential reward for being close to target horizontally
    #     position_reward = 0.2 * np.exp(-horizontal_distance / 10.0)
    
    #     if horizontal_distance < self.landing_zone_radius:
    #         position_reward += 0.15
    #     # If in zone and low altitude, LAND NOW!
    #         if altitude < 3.9:
    #             position_reward += 0.5
    
    #     reward += position_reward
        
    #     # 3. ORIENTATION REWARD (Critical: must stay upright)
    #     # -------------------------------------------------------------------------
    #     # Strong reward for being upright
    #     orientation_reward = 0.15 * upright_score
        
    #     # Penalty for tilting
    #     if tilt_angle > 0.4:  # More than ~17 degrees
    #         orientation_reward -= 0.2 * (tilt_angle - 0.4)
        
    #     # Penalty for spinning
    #     if angular_speed > 0.5:
    #         orientation_reward -= 0.1 * angular_speed
        
    #     reward += orientation_reward
        
    #     # 4. STABILITY REWARD (Encourage smooth, controlled flight)
    #     # -------------------------------------------------------------------------
    #     stability_reward = 0.0
        
    #     # Reward low angular velocity (stable flight)
    #     if angular_speed < 0.2:
    #         stability_reward += 0.1
        
    #     # Reward being in a "good state" for landing
    #     if altitude < 10.0:
    #         good_state = (upright_score > 0.9 and 
    #                      horizontal_distance < self.landing_zone_radius * 1.5 and
    #                      vertical_speed < 4.0)
    #         if good_state:
    #             stability_reward += 0.2
        
    #     reward += stability_reward
        
    #     # 5. FUEL MANAGEMENT (Gentle penalty for fuel usage)
    #     # -------------------------------------------------------------------------
    #     # Small penalty for low fuel (but don't discourage necessary thrust)
    #     fuel_penalty_factor = 0.5 + (self.step_count / self.max_steps) * 2.0
    
    #     if fuel_fraction < 0.3:
    #         reward -= (0.3 - fuel_fraction) * fuel_penalty_factor
    
    #     if self.fuel_remaining <= 0:
    #         reward -= 2.0
        
    #     #hovering detection:
    #     is_hovering = (altitude < 10.0 and 
    #                abs(vz) < 1.0 and 
    #                horizontal_distance < self.landing_zone_radius * 1.5)
    
    #     if is_hovering:
    #         hover_penalty = -0.3 - (self.step_count / self.max_steps) * 0.5
    #         reward += hover_penalty
        
    #     if altitude < 5.0:
    #         reward -= 0.5


    #     # 6. PROGRESS REWARDS (Encourage improvement over time)
    #     # -------------------------------------------------------------------------
    #     progress_reward = 0.0
        
    #     # Reward getting closer to target
    #     if self.previous_distance is not None:
    #         distance_improvement = self.previous_distance - horizontal_distance
    #         progress_reward += distance_improvement * 0.5  # Scale appropriately
    #     self.previous_distance = horizontal_distance
        
    #     # Reward altitude progress (when appropriate)
    #     if self.previous_altitude is not None:
    #         if altitude > 5.0:
    #             # Reward descent at high altitude
    #             altitude_progress = self.previous_altitude - altitude
    #             if altitude_progress > 0:  # Descending
    #                 progress_reward += altitude_progress * 0.1
    #     self.previous_altitude = altitude
        
    #     reward += progress_reward
        
    #     # 7. TIME PENALTY (Encourage efficiency, but keep small)
    #     # -------------------------------------------------------------------------
    #       # Small penalty to encourage not wasting time
        

    #     #8. Descent reward :
    #     if vz < -1.0 and altitude > 3.0:
    #         descent_reward = min(abs(vz) * 0.1, 0.3)
    #         reward += descent_reward

    #     # ============================================================================
    #     # CONSTRAINT PENALTIES (Prevent bad behavior)
    #     # ============================================================================
        
    #     # Strong penalty for dangerous situations
    #     if altitude < 3.0 and vertical_speed > 8.0:
    #         reward -= 1.5  # About to crash hard!
        
    #     if altitude < 10.0 and upright_score < 0.5:
    #         reward -= 0.8  # Tilted too much near ground
        
    #     if horizontal_distance > 20.0:
    #         reward -= 0.5  # Too far from target
        
    #     # ============================================================================
    #     # REWARD CLIPPING (Keep step rewards in reasonable range)
    #     # ============================================================================
    #     # Clip step rewards to prevent extreme values (but allow terminal rewards through)
    #     reward = np.clip(reward, -3.0, 3.0)
        
    #     return reward    
    
    def _calculate_reward(self, obs, landed, terminated):
        """
        Anti-hovering reward function.
        Key changes:
        1. Strong time penalty to discourage hovering
        2. Exponential altitude rewards (gets urgent near ground)
        3. Fuel penalty increases over time
        4. Landing bonus >>> accumulated step rewards
        """

        # Parse observation
        pos = obs[0:3]
        vel = obs[3:6]
        quat = obs[6:10]
        ang_vel = obs[10:13]
        fuel_fraction = obs[13]

        # Calculate key metrics
        x, y, z = pos
        vx, vy, vz = vel
        altitude = z

        horizontal_distance = np.sqrt((x - self.target_x)**2 + (y - self.target_y)**2)
        speed = np.linalg.norm(vel)
        vertical_speed = abs(vz)
        horizontal_speed = np.sqrt(vx**2 + vy**2)
        angular_speed = np.linalg.norm(ang_vel)

        # Calculate upright score
        rotation_matrix = np.array(p.getMatrixFromQuaternion(quat)).reshape(3, 3)
        upright_score = rotation_matrix[2, 2]
        tilt_angle = np.arccos(np.clip(upright_score, -1, 1))

        reward = 0.0

        # ========================================================================
        # TERMINAL REWARDS
        # ========================================================================

        if landed:
            # Massive landing bonus (must be >> accumulated step rewards)
            base_reward = 2000.0

            # Distance bonus
            distance_bonus = 1000.0 * np.exp(-horizontal_distance / 1.5)

            # Speed bonus (gentle landing)
            speed_bonus = 500.0 * np.exp(-speed / 2.0)

            # Orientation bonus
            orientation_bonus = 300.0 * upright_score

            # Fuel bonus (reward efficiency)
            fuel_bonus = 500.0 * fuel_fraction

            # Time bonus (reward fast landing)
            time_bonus = 500.0 * (1.0 - self.step_count / self.max_steps)

            total_landing_reward = (base_reward + distance_bonus + speed_bonus + 
                                   orientation_bonus + fuel_bonus + time_bonus)

            print(f"🚀 LANDED! Reward: {total_landing_reward:.0f} "
                  f"(dist: {horizontal_distance:.2f}m, speed: {speed:.2f}m/s, steps: {self.step_count})")

            return total_landing_reward

        if terminated and not landed:
            # Crash penalty
            base_penalty = -800.0

            # Less penalty if close and trying
            if horizontal_distance < self.landing_zone_radius:
                base_penalty *= 0.6
            if upright_score > 0.8:
                base_penalty *= 0.8

            print(f"💥 CRASHED! Penalty: {base_penalty:.0f}")
            return base_penalty

        # ========================================================================
        # STEP REWARDS - Designed to prevent hovering
        # ========================================================================

        # 1. STRONG TIME PENALTY (discourages hovering)
        # -----------------------------------------------------------------------
        # Penalty increases over time - hovering becomes more expensive
        time_pressure = 0.05 + (self.step_count / self.max_steps) * 0.2
        reward -= time_pressure

        # 2. ALTITUDE REWARD (exponentially increasing urgency to land)
        # -----------------------------------------------------------------------
        # The lower you are, the more urgent it is to land NOW
        if altitude > 20.0:
            altitude_reward = -0.1  # Penalty for being high
            if vz < -2.0:  # Reward descending
                altitude_reward += 0.15
        elif altitude > 10.0:
            # Mid altitude - start being urgent
            altitude_reward = 0.2 - altitude * 0.01
            if vz < -2.0:
                altitude_reward += 0.2
        elif altitude > 3.0:
            # Low altitude - VERY urgent to land
            altitude_reward = 0.5 - altitude * 0.05
            # Strong reward for descending
            if vz < -1.0:
                altitude_reward += 0.4
            # Penalty for hovering (not descending)
            if abs(vz) < 0.5:
                altitude_reward -= 0.5  # HOVERING PENALTY!
        else:
            # Very close to ground - just land!
            altitude_reward = 0.8
            if abs(vz) < 0.2:  # Hovering at ground level
                altitude_reward -= 1.0  # SEVERE HOVERING PENALTY!

        reward += altitude_reward

        # 3. POSITION REWARD (be above landing zone)
        # -----------------------------------------------------------------------
        position_reward = 0.2 * np.exp(-horizontal_distance / 10.0)

        if horizontal_distance < self.landing_zone_radius:
            position_reward += 0.15
            # If in zone and low altitude, LAND NOW!
            if altitude < 4.5:
                position_reward += 0.3

        reward += position_reward

        # 4. ORIENTATION REWARD (stay upright)
        # -----------------------------------------------------------------------
        orientation_reward = 0.15 * upright_score  # Reduced from 0.3

        if tilt_angle > 0.4:
            orientation_reward -= 0.2 * (tilt_angle - 0.4)

        if angular_speed > 0.5:
            orientation_reward -= 0.1 * angular_speed

        reward += orientation_reward

        # 5. FUEL PENALTY (increases over time)
        # -----------------------------------------------------------------------
        # Penalize fuel usage more as time goes on
        fuel_penalty_factor = 0.5 + (self.step_count / self.max_steps) * 2.0

        if fuel_fraction < 0.3:
            reward -= (0.3 - fuel_fraction) * fuel_penalty_factor

        if self.fuel_remaining <= 0:
            reward -= 2.0  # Severe penalty for running out

        # 6. HOVERING DETECTION (explicit penalty)
        # -----------------------------------------------------------------------
        # Detect hovering: low altitude + low vertical speed + in zone
        is_hovering = (altitude < 10.0 and 
                       abs(vz) < 1.0 and 
                       horizontal_distance < self.landing_zone_radius * 1.5)

        if is_hovering:
            # Hovering penalty increases the longer you hover
            hover_penalty = -0.3 - (self.step_count / self.max_steps) * 0.5
            reward += hover_penalty

            # Extra penalty if hovering at very low altitude
            if altitude < 5.0:
                reward -= 0.5

        # 7. DESCENT REWARD (reward active descent)
        # -----------------------------------------------------------------------
        if vz < -1.0 and altitude > 3.0:
            # Reward descending (but not too fast)
            descent_reward = min(abs(vz) * 0.1, 0.3)
            reward += descent_reward

        # ========================================================================
        # CONSTRAINT PENALTIES
        # ========================================================================

        # Dangerous situations
        if altitude < 3.0 and vertical_speed > 8.0:
            reward -= 1.5

        if altitude < 10.0 and upright_score < 0.5:
            reward -= 0.8

        if horizontal_distance > 20.0:
            reward -= 0.5

        # Clip to reasonable range
        reward = np.clip(reward, -3.0, 3.0)

        return reward
    
    def _check_termination(self, obs):
        """Check if the episode should terminate with stricter conditions."""
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
            if self.difficulty=='easy':
                is_on_target = horizontal_distance < self.landing_zone_radius
                is_slow_enough = speed < 6  # Stricter speed requirement
                is_upright = upright_score > 0.97
            elif self.difficulty=='medium':
                is_on_target = horizontal_distance < self.landing_zone_radius
                is_slow_enough = speed < 5  # Stricter speed requirement
                is_upright = upright_score > 0.98
            else:
                is_on_target = horizontal_distance < self.landing_zone_radius
                is_slow_enough = speed < 4.5  # Stricter speed requirement
                is_upright = upright_score > 0.98
                

             # Stricter upright requirement (less than ~18 deg tilt)

            if is_on_target and is_slow_enough and is_upright:
                # time.sleep(2)
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
        if self.difficulty=='easy':
            if horizontal_distance > 10.0 or altitude > 27.0 or altitude < -2.0:
                print("🚫 OUT OF BOUNDS!")
                terminated = True
                return terminated, truncated, landed
        elif self.difficulty=='medium':
            if horizontal_distance > 10.0 or altitude > 44.0 or altitude < -2.0:
                print("🚫 OUT OF BOUNDS!")
                terminated = True
                return terminated, truncated, landed
        else:
            if horizontal_distance > 10.0 or altitude > 62.0 or altitude < -2.0:
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
                
                # Draw thruster positions for debugging
                rotation_matrix = np.array(p.getMatrixFromQuaternion(p.getBasePositionAndOrientation(self.rocket)[1])).reshape(3, 3)
                
                for name, local_pos in self.thruster_positions.items():
                    world_pos = pos + rotation_matrix @ local_pos
                    # Draw small sphere at thruster location
                    p.addUserDebugLine(
                        world_pos, 
                        [world_pos[0], world_pos[1], world_pos[2] + 0.2],
                        [0, 1, 1], lineWidth=2, lifeTime=0.1
                    )

    def close(self):
        """Clean up resources"""
        if self.client is not None:
            p.disconnect(self.client)
            self.client = None


