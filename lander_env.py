import os
import time
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import pybullet as p
import pybullet_data

class Falcon9LandingEnv(gym.Env):
    """
    Falcon 9 landing environment with PROPER curriculum learning.
    
    CRITICAL FIX: Reward scaling compatible with VecNormalize!
    - Step rewards: -1 to +1 (for smooth shaping)
    - Terminal rewards: -10 to +100 (still significant after normalization)
    """
    
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 50,
    }

    def __init__(self, render_mode=None, difficulty='easy', curriculum_level=1,
                 rocket_urdf_path=r"C:\Users\Lenovo\Desktop\falcon9_project\assets\rocket.urdf", 
                 landing_pad_urdf_path=r"C:\Users\Lenovo\Desktop\falcon9_project\assets\landing_pad.urdf"):
        super().__init__()
        self.render_mode = render_mode
        self.difficulty = difficulty
        self.curriculum_level = curriculum_level  # 1-5, increasing difficulty
        
        # Action space
        self.action_space = spaces.Box(
            low=np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32
        )
        
        # Observation space
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(14,), dtype=np.float32
        )
        
        # Physics parameters
        self.max_main_thrust = 130000.0
        self.max_thruster_force = 5000.0
        self.rocket_mass = 1500.0
        self.initial_fuel = 1000.0
        self.fuel_consumption_rate = 7.5
        
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
        self.landing_zone_radius = 10.0
        
        # Tracking variables - PROPERLY INITIALIZED
        self.previous_distance = None
        self.previous_altitude = None
        self.initial_distance = None
        self.initial_altitude = None
        
        # Success tracking
        self.episode_count = 0
        self.success_count = 0
        
        # Thruster positions
        self.thruster_positions = {
            'north': [0, 2.5, 2.0],
            'east':  [2.5, 0, 2.0],
            'south': [0, -2.5, 2.0],
            'west':  [-2.5, 0, 2.0]
        }
        
        self._setup_urdf_paths(rocket_urdf_path, landing_pad_urdf_path)
        self._set_curriculum_parameters()

    def _set_curriculum_parameters(self):
        """Set difficulty parameters based on curriculum level"""
        if self.curriculum_level == 1:
            # Level 1: Very easy - learn to use thrust and stay upright
            self.start_altitude_range = (10.0, 15.0)
            self.start_position_range = (-0.5, 0.5)
            self.start_velocity_range = (-0.5, -0.2)
            self.landing_zone_radius = 15.0
            self.max_tilt_for_landing = 0.85  # ~32 degrees
            self.max_speed_for_landing = 10.0
            self.altitude_bounds = (-2.0, 20.0)
            self.horizontal_bounds = 20.0
            
        elif self.curriculum_level == 2:
            # Level 2: Easy - learn to descend and position
            self.start_altitude_range = (15.0, 20.0)
            self.start_position_range = (-1.0, 1.0)
            self.start_velocity_range = (-1.0, -0.5)
            self.landing_zone_radius = 12.0
            self.max_tilt_for_landing = 0.90  # ~26 degrees
            self.max_speed_for_landing = 8.0
            self.altitude_bounds = (-2.0, 25.0)
            self.horizontal_bounds = 20.0
            
        elif self.curriculum_level == 3:
            # Level 3: Medium - learn controlled landing
            self.start_altitude_range = (20.0, 30.0)
            self.start_position_range = (-2.0, 2.0)
            self.start_velocity_range = (-1.5, -0.8)
            self.landing_zone_radius = 10.0
            self.max_tilt_for_landing = 0.93  # ~21 degrees
            self.max_speed_for_landing = 6.0
            self.altitude_bounds = (-2.0, 35.0)
            self.horizontal_bounds = 20.0
            
        elif self.curriculum_level == 4:
            # Level 4: Hard - precise landing
            self.start_altitude_range = (30.0, 40.0)
            self.start_position_range = (-2.5, 2.5)
            self.start_velocity_range = (-2.0, -1.0)
            self.landing_zone_radius = 8.0
            self.max_tilt_for_landing = 0.95  # ~18 degrees
            self.max_speed_for_landing = 5.0
            self.altitude_bounds = (-2.0, 45.0)
            self.horizontal_bounds = 20.0
            
        else:  # Level 5
            # Level 5: Expert - realistic conditions
            self.start_altitude_range = (40.0, 50.0)
            self.start_position_range = (-3.0, 3.0)
            self.start_velocity_range = (-2.5, -1.5)
            self.landing_zone_radius = 7.0
            self.max_tilt_for_landing = 0.97  # ~14 degrees
            self.max_speed_for_landing = 4.0
            self.altitude_bounds = (-2.0, 55.0)
            self.horizontal_bounds = 20.0

    def set_curriculum_level(self, level):
        """Update curriculum level (called by training script)"""
        self.curriculum_level = max(1, min(5, level))
        self._set_curriculum_parameters()

    def _setup_urdf_paths(self, rocket_path=None, pad_path=None):
        """Setup paths to URDF files"""
        if rocket_path is not None:
            if os.path.exists(rocket_path):
                self.rocket_urdf_path = os.path.abspath(rocket_path)
            else:
                raise FileNotFoundError(f"Rocket URDF not found at: {rocket_path}")
        
        if pad_path is not None:
            if os.path.exists(pad_path):
                self.pad_urdf_path = os.path.abspath(pad_path)
            else:
                raise FileNotFoundError(f"Landing pad URDF not found at: {pad_path}")

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
            fixedTimeStep=1.0/240.0,
            numSolverIterations=20
        )
        p.setRealTimeSimulation(0)

    def reset(self, seed=None, options=None):
        """Reset the environment"""
        super().reset(seed=seed)
        
        self._connect_physics()
        p.resetSimulation()
        p.setGravity(0, 0, -9.81)
        
        # Load ground and landing pad
        self.ground = p.loadURDF("plane.urdf")
        self.landing_pad = p.loadURDF(
            self.pad_urdf_path,
            basePosition=[self.target_x, self.target_y, 0.0],
            useFixedBase=True
        )
        
        # Random initial conditions based on curriculum
        if seed is not None:
            np.random.seed(seed)
        
        start_x = np.random.uniform(*self.start_position_range)
        start_y = np.random.uniform(*self.start_position_range)
        start_z = np.random.uniform(*self.start_altitude_range)
        
        # Small random orientation
        roll = np.random.uniform(-0.05, 0.05)
        pitch = np.random.uniform(-0.05, 0.05)
        yaw = np.random.uniform(-np.pi, np.pi)
        start_orientation = p.getQuaternionFromEuler([roll, pitch, yaw])
        
        self.rocket = p.loadURDF(
            self.rocket_urdf_path,
            basePosition=[start_x, start_y, start_z],
            baseOrientation=start_orientation,
            useFixedBase=False
        )
        
        # Initial velocity
        initial_vel = [
            np.random.uniform(-0.5, 0.5),
            np.random.uniform(-0.5, 0.5),
            np.random.uniform(*self.start_velocity_range)
        ]
        p.resetBaseVelocity(self.rocket, linearVelocity=initial_vel)
        
        # Reset state
        self.step_count = 0
        self.fuel_remaining = self.initial_fuel
        self.episode_count += 1
        
        # Material properties
        p.changeDynamics(self.rocket, -1, linearDamping=0.02)
        p.changeDynamics(self.landing_pad, -1, restitution=0.3, lateralFriction=1.0)
        
        # Get initial observation and initialize tracking
        observation = self._get_observation()
        pos = observation[0:3]
        self.initial_distance = np.sqrt((pos[0] - self.target_x)**2 + (pos[1] - self.target_y)**2)
        self.initial_altitude = pos[2]
        self.previous_distance = self.initial_distance
        self.previous_altitude = self.initial_altitude
        
        info = self._get_info()
        info['curriculum_level'] = self.curriculum_level
        
        return observation, info

    def _get_observation(self):
        """Get current observation"""
        position, orientation = p.getBasePositionAndOrientation(self.rocket)
        velocity, angular_velocity = p.getBaseVelocity(self.rocket)
        
        fuel_fraction = self.fuel_remaining / self.initial_fuel
        
        observation = np.array([
            position[0], position[1], position[2],
            velocity[0], velocity[1], velocity[2],
            orientation[0], orientation[1], orientation[2], orientation[3],
            angular_velocity[0], angular_velocity[1], angular_velocity[2],
            fuel_fraction
        ], dtype=np.float32)
        
        return observation

    def _get_info(self):
        """Get additional info"""
        position, orientation = p.getBasePositionAndOrientation(self.rocket)
        velocity, _ = p.getBaseVelocity(self.rocket)
        
        distance_to_target = np.sqrt(
            (position[0] - self.target_x)**2 + 
            (position[1] - self.target_y)**2
        )
        
        euler = p.getEulerFromQuaternion(orientation)
        
        return {
            "distance_to_target": distance_to_target,
            "altitude": position[2],
            "speed": np.linalg.norm(velocity),
            "fuel_remaining": self.fuel_remaining,
            "euler_angles": euler,
            "step_count": self.step_count,
            "curriculum_level": self.curriculum_level
        }

    def _create_engine_particles(self, position, orientation, thrust_intensity):
        """Create particle effects for main engine"""
        rotation_matrix = np.array(p.getMatrixFromQuaternion(orientation)).reshape(3, 3)
        exhaust_direction = -rotation_matrix[:, 2]
        engine_local_pos = [0, 0, -3.8]
        engine_world_pos = position + rotation_matrix @ engine_local_pos
        
        num_particles = int(thrust_intensity * 4)
        
        for i in range(num_particles):
            spread = 0.1 + thrust_intensity * 0.5
            random_offset = np.random.uniform(-spread, spread, 3)
            random_offset[2] *= 0.3
            
            particle_start = engine_world_pos + random_offset
            exhaust_length = 3.0 + thrust_intensity * 2.0
            particle_end = particle_start + exhaust_direction * exhaust_length + random_offset * 0.5
            
            if thrust_intensity < 0.3:
                color = [0.3, 0.5, 1.0]
            elif thrust_intensity < 0.7:
                color = [0.8, 0.6, 0.2]
            else:
                color = [1.0, 0.3, 0.1]
            
            color = [c + np.random.uniform(-0.1, 0.1) for c in color]
            color = [max(0, min(1, c)) for c in color]
            
            p.addUserDebugLine(
                particle_start, particle_end, lineColorRGB=color,
                lineWidth=2.0 + thrust_intensity * 2.0, lifeTime=0.05
            )
    
    def _create_thruster_particles(self, position, orientation, thruster_forces):
        """Create particle effects for thrusters"""
        rotation_matrix = np.array(p.getMatrixFromQuaternion(orientation)).reshape(3, 3)
        thruster_names = ['north', 'east', 'south', 'west']
        
        for thruster_name, force in zip(thruster_names, thruster_forces):
            if force > 0.05:
                local_pos = self.thruster_positions[thruster_name]
                world_pos = position + rotation_matrix @ local_pos
                
                if thruster_name == 'north':
                    exhaust_dir = rotation_matrix[:, 1]
                elif thruster_name == 'south':
                    exhaust_dir = -rotation_matrix[:, 1]
                elif thruster_name == 'east':
                    exhaust_dir = rotation_matrix[:, 0]
                elif thruster_name == 'west':
                    exhaust_dir = -rotation_matrix[:, 0]
                
                num_particles = int(force * 3)
                
                for j in range(num_particles):
                    particle_start = world_pos + np.random.uniform(-0.1, 0.1, 3)
                    exhaust_length = 0.5 + force * 1.5
                    particle_end = particle_start + exhaust_dir * exhaust_length
                    color = [0.7, 0.8, 1.0]
                    
                    p.addUserDebugLine(
                        particle_start, particle_end, lineColorRGB=color,
                        lineWidth=1.5, lifeTime=0.04
                    )

    def step(self, action):
        """Execute one time step"""
        
        if self.render_mode == "human":
            r_position, _ = p.getBasePositionAndOrientation(self.rocket)
            if r_position[2] < 35:
                p.resetDebugVisualizerCamera(
                    cameraDistance=20, cameraYaw=0, cameraPitch=-5,
                    cameraTargetPosition=[0, 0, 5]
                )
            else:
                p.resetDebugVisualizerCamera(
                    cameraDistance=15, cameraYaw=10, cameraPitch=-40,
                    cameraTargetPosition=r_position
                )

        action = np.clip(action, self.action_space.low, self.action_space.high)
        
        main_thrust = action[0]
        thruster_forces = action[1:5]
        
        if self.fuel_remaining <= 0:
            main_thrust = 0.0
            thruster_forces = np.zeros(4)
        
        # Apply main engine thrust
        if main_thrust > 0.01:
            thrust_force = main_thrust * self.max_main_thrust
            position, orientation = p.getBasePositionAndOrientation(self.rocket)
            
            rotation_matrix = np.array(p.getMatrixFromQuaternion(orientation)).reshape(3, 3)
            thrust_direction = rotation_matrix[:, 2]
            thrust_vector = thrust_direction * thrust_force
            thrust_point = [0, 0, -4.0]
            
            p.applyExternalForce(self.rocket, -1, thrust_vector, thrust_point, p.LINK_FRAME)
            
            if self.render_mode == "human":
                self._create_engine_particles(position, orientation, main_thrust)
            
            fuel_used = main_thrust * self.fuel_consumption_rate * (1.0/60.0)
            self.fuel_remaining = max(0, self.fuel_remaining - fuel_used)
        
        # Apply thrusters
        position, orientation = p.getBasePositionAndOrientation(self.rocket)
        rotation_matrix = np.array(p.getMatrixFromQuaternion(orientation)).reshape(3, 3)
        
        thruster_names = ['north', 'east', 'south', 'west']
        force_directions = [
            -rotation_matrix[:, 1],  # north
            -rotation_matrix[:, 0],  # east
            rotation_matrix[:, 1],   # south
            rotation_matrix[:, 0]    # west
        ]
        
        for i, (thruster_name, thrust_level, force_dir) in enumerate(zip(thruster_names, thruster_forces, force_directions)):
            if thrust_level > 0.01:
                force_magnitude = thrust_level * self.max_thruster_force
                local_pos = self.thruster_positions[thruster_name]
                force_vector = force_dir * force_magnitude
                
                p.applyExternalForce(self.rocket, -1, force_vector, local_pos, p.LINK_FRAME)
                
                fuel_used = thrust_level * 0.2 * (1.0/60.0)
                self.fuel_remaining = max(0, self.fuel_remaining - fuel_used)
        
        if self.render_mode == "human" and any(f > 0.05 for f in thruster_forces):
            self._create_thruster_particles(position, orientation, thruster_forces)
        
        # Step simulation
        for _ in range(4):
            p.stepSimulation()
        
        self.step_count += 1
        
        # Get results
        observation = self._get_observation()
        terminated, truncated, landed = self._check_termination(observation)
        reward = self._calculate_reward(observation, landed, terminated)
        info = self._get_info()
        info["landed_successfully"] = landed
        
        if landed:
            self.success_count += 1
        
        if self.render_mode == "human":
            time.sleep(1.0/60.0)
        
        return observation, reward, terminated, truncated, info

    def _calculate_reward(self, obs, landed, terminated):
        """
        CRITICAL FIX: Reward scaled for VecNormalize compatibility!
        - Step rewards: -1 to +1
        - Terminal rewards: -10 to +100
        """
        
        pos = obs[0:3]
        vel = obs[3:6]
        quat = obs[6:10]
        fuel_fraction = obs[13]
        
        x, y, z = pos
        vx, vy, vz = vel
        
        horizontal_distance = np.sqrt((x - self.target_x)**2 + (y - self.target_y)**2)
        speed = np.linalg.norm(vel)
        vertical_speed = abs(vz)
        
        # Upright score
        rotation_matrix = np.array(p.getMatrixFromQuaternion(quat)).reshape(3, 3)
        upright_score = rotation_matrix[2, 2]
        
        # ========================================================================
        # TERMINAL REWARDS (scaled to survive VecNormalize)
        # ========================================================================
        
        if landed:
            # Base landing reward
            reward = 100.0
            
            # Precision bonus
            if horizontal_distance < 2.0:
                reward += 20.0
            elif horizontal_distance < 5.0:
                reward += 10.0
            
            # Gentleness bonus
            if speed < 2.0:
                reward += 15.0
            elif speed < 4.0:
                reward += 8.0
            
            # Orientation bonus
            if upright_score > 0.98:
                reward += 10.0
            elif upright_score > 0.95:
                reward += 5.0
            
            # Fuel efficiency
            reward += fuel_fraction * 10.0
            
            # Time bonus
            reward += (1.0 - self.step_count / self.max_steps) * 20.0
            
            return reward
        
        if terminated and not landed:
            # Crash penalty
            penalty = -10.0
            
            # Less harsh if you were trying
            if horizontal_distance < self.landing_zone_radius and upright_score > 0.7:
                penalty = -5.0
            
            return penalty
        
        # ========================================================================
        # STEP REWARDS (small, for shaping only)
        # ========================================================================
        
        reward = 0.0
        
        # 1. Altitude progress
        altitude_progress = 0.0
        if z > 5.0:
            # High up: reward descending
            if vz < -0.5:
                altitude_progress = 0.3
        elif z > 2.0:
            # Low: reward being slow
            if abs(vz) < 3.0:
                altitude_progress = 0.4
            if abs(vz) < 1.5:
                altitude_progress += 0.3
        else:
            # Very low: land now!
            if abs(vz) < 2.0:
                altitude_progress = 0.6
        
        reward += altitude_progress
        
        # 2. Position reward
        distance_reward = 0.3 * np.exp(-horizontal_distance / 10.0)
        if horizontal_distance < self.landing_zone_radius:
            distance_reward += 0.3
        reward += distance_reward
        
        # 3. Orientation reward
        orientation_reward = 0.2 * upright_score
        if upright_score < 0.6:
            orientation_reward -= 0.3
        reward += orientation_reward
        
        # 4. Progress rewards
        if self.previous_distance is not None:
            distance_improvement = (self.previous_distance - horizontal_distance) / self.initial_distance
            reward += distance_improvement * 2.0
        
        if self.previous_altitude is not None and z > 5.0:
            altitude_improvement = (self.previous_altitude - z) / self.initial_altitude
            if altitude_improvement > 0:
                reward += altitude_improvement * 1.0
        
        self.previous_distance = horizontal_distance
        self.previous_altitude = z
        
        # 5. Small penalties
        reward -= 0.01  # Time penalty
        
        if fuel_fraction < 0.15:
            reward -= 0.2
        
        if self.fuel_remaining <= 0:
            reward -= 1.0
        
        # Safety penalties
        if z < 3.0 and speed > 8.0:
            reward -= 1.0
        
        if z < 10.0 and upright_score < 0.5:
            reward -= 0.5
        
        # Clip
        reward = np.clip(reward, -2.0, 2.0)
        
        return reward

    def _check_termination(self, obs):
        """Check termination"""
        pos = obs[0:3]
        vel = obs[3:6]
        quat = obs[6:10]
        
        altitude = pos[2]
        speed = np.linalg.norm(vel)
        horizontal_distance = np.sqrt(
            (pos[0] - self.target_x)**2 + (pos[1] - self.target_y)**2
        )
        
        rotation_matrix = np.array(p.getMatrixFromQuaternion(quat)).reshape(3, 3)
        upright_score = rotation_matrix[2, 2]
        
        terminated = False
        truncated = False
        landed = False
        
        # Landing check
        if altitude <= 2.5:
            is_on_target = horizontal_distance < self.landing_zone_radius
            is_slow_enough = speed < self.max_speed_for_landing
            is_upright = upright_score > self.max_tilt_for_landing
            
            if is_on_target and is_slow_enough and is_upright:
                terminated = True
                landed = True
                return terminated, truncated, landed
            else:
                terminated = True
                landed = False
                return terminated, truncated, landed
        
        # Out of bounds
        if (horizontal_distance > self.horizontal_bounds or 
            altitude > self.altitude_bounds[1] or 
            altitude < self.altitude_bounds[0]):
            terminated = True
            return terminated, truncated, landed
        
        # Time limit
        if self.step_count >= self.max_steps:
            truncated = True
            return terminated, truncated, landed
        
        # Fuel depletion
        if self.fuel_remaining <= 0 and altitude > 1.0:
            terminated = True
            return terminated, truncated, landed
        
        return terminated, truncated, landed

    def render(self):
        """Render the environment"""
        if self.render_mode == "human" and hasattr(self, 'rocket') and self.rocket is not None:
            pos, _ = p.getBasePositionAndOrientation(self.rocket)
            info = self._get_info()
            
            debug_text = f"Alt: {pos[2]:.1f}m | Dist: {info['distance_to_target']:.1f}m | Fuel: {self.fuel_remaining:.0f}kg | Level: {self.curriculum_level}"
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