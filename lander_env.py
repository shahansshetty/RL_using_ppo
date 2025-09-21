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

    def __init__(self, render_mode="human", rocket_urdf_path=r"C:\Users\Lenovo\Desktop\falcon9_project\assets\rocket.urdf", landing_pad_urdf_path=r"C:\Users\Lenovo\Desktop\falcon9_project\assets\landing_pad.urdf"):
        super().__init__()
        self.render_mode = render_mode
        
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
        self.max_main_thrust = 100000.0  # Newtons
        self.max_thruster_force = 5000.0  # Newtons per individual thruster
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
        roll = np.random.uniform(-0.5, 0.5)
        pitch = np.random.uniform(-0.5, 0.5) 
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
            landing_bonus = 500.0
            if horizontal_distance < 1.0:
                landing_bonus += 500.0  # Perfect landing
            if horizontal_distance < 2.0:
                landing_bonus += 400.0  # Perfect landing
            if horizontal_distance < 3.0:
                landing_bonus += 300.0  # Perfect landing
            elif horizontal_distance < 4:
                landing_bonus += 200.0  # Good landing
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
            is_on_target = horizontal_distance < self.landing_zone_radius
            is_slow_enough = speed < 4.5  # Stricter speed requirement
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


# Example usage and testing
# if __name__ == "__main__":
#     print("Falcon9 Environment with 4 Individual Thrusters Test")
#     print("="*55)
    
#     # Test the new 4-thruster system
#     env = Falcon9LandingEnv(render_mode="human")
    
#     obs, info = env.reset()
#     print(f"Initial observation shape: {obs.shape}")
#     print(f"Action space: {env.action_space}")
#     print(f"Action space shape: {env.action_space.shape}")
#     print("Actions: [main_thrust, thruster_north, thruster_east, thruster_south, thruster_west]")
    
#     for step in range(500):
#         # Test control using individual thrusters
#         pos = obs[0:3]
#         vel = obs[3:6]
#         quat = obs[6:10]
#         ang_vel = obs[10:13]
        
#         # Basic landing controller with individual thruster control
#         altitude_error = pos[2] - 1.0  # Target 1m altitude
#         vertical_vel = vel[2]
        
#         # Main thrust control
#         main_thrust = 0.4 + altitude_error * 0.05 - vertical_vel * 0.1
#         main_thrust = np.clip(main_thrust, 0.0, 1.0)
        
#         # Individual thruster control for attitude and position
#         rotation_matrix = np.array(p.getMatrixFromQuaternion(quat)).reshape(3, 3)
        
#         # Calculate desired corrections
#         tilt_x = rotation_matrix[0, 2]  # Roll tilt
#         tilt_y = rotation_matrix[1, 2]  # Pitch tilt
        
#         # Horizontal position errors
#         pos_error_x = pos[0] - 0.0  # Target x=0
#         pos_error_y = pos[1] - 0.0  # Target y=0
        
#         # Simple thruster control logic
#         # North thruster (front): controls pitch and Y movement
#         thruster_north = np.clip(-tilt_y * 0.5 - pos_error_y * 0.1 - ang_vel[1] * 0.1, 0.0, 1.0)
        
#         # South thruster (back): opposite of north
#         thruster_south = np.clip(tilt_y * 0.5 + pos_error_y * 0.1 + ang_vel[1] * 0.1, 0.0, 1.0)
        
#         # East thruster (right): controls roll and X movement
#         thruster_east = np.clip(-tilt_x * 0.5 - pos_error_x * 0.1 - ang_vel[0] * 0.1, 0.0, 1.0)
        
#         # West thruster (left): opposite of east
#         thruster_west = np.clip(tilt_x * 0.5 + pos_error_x * 0.1 + ang_vel[0] * 0.1, 0.0, 1.0)
        
#         action = np.array([main_thrust, thruster_north, thruster_east, thruster_south, thruster_west])
        
#         obs, reward, terminated, truncated, info = env.step(action)
        
#         if step % 60 == 0:  # Print every second
#             print(f"Step {step}: Alt={pos[2]:.1f}m, Reward={reward:.1f}")
#             print(f"  Thrusters - N:{thruster_north:.2f}, E:{thruster_east:.2f}, S:{thruster_south:.2f}, W:{thruster_west:.2f}")
        
#         if terminated or truncated:
#             print(f"Episode ended at step {step}")
#             print(f"Final distance to target: {info['distance_to_target']:.2f}m")
#             print(f"Final altitude: {info['altitude']:.2f}m")
#             print(f"Final speed: {info['speed']:.2f}m/s")
#             print(f"Landing successful: {info.get('landed_successfully', False)}")
#             break
    
#     env.close()
#     print("4-Thruster environment test completed!")