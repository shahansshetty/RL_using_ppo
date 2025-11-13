"""
Real-Time Neural Network Dashboard

Live dashboard showing:
- Network activations in real-time
- Rocket state and actions
- Decision-making process
- Performance metrics

Great for presentations and understanding network behavior!
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.gridspec import GridSpec
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from lander_env import Falcon9LandingEnv
import os

class LiveNeuralDashboard:
    """Real-time dashboard for neural network visualization"""
    
    def __init__(self, model_path, vecnorm_path=None):
        print("Loading model for live dashboard...")
        self.model = PPO.load(model_path)
        
        def make_env():
            return Falcon9LandingEnv(render_mode=None, curriculum_level=5)
        
        self.env = DummyVecEnv([make_env])
        
        if vecnorm_path and os.path.exists(vecnorm_path):
            self.env = VecNormalize.load(vecnorm_path, self.env)
            self.env.training = False
            self.env.norm_reward = False
        
        self.policy = self.model.policy
        
        # State tracking
        self.obs_history = []
        self.action_history = []
        self.reward_history = []
        self.value_history = []
        self.max_history = 100
        
        # Reset environment
        self.obs = self.env.reset()
        self.episode_reward = 0
        self.step_count = 0
        
        print("✅ Dashboard ready!")
    
    def get_network_data(self, obs):
        """Get all network outputs and intermediate values"""
        with torch.no_grad():
            obs_tensor = torch.FloatTensor(obs).to(self.model.device)
            
            # Get action
            action, _ = self.model.predict(obs, deterministic=False)
            
            # Get value estimate
            features = self.policy.extract_features(obs_tensor)
            if hasattr(self.policy, 'mlp_extractor'):
                value_features = self.policy.mlp_extractor.value_net(features)
                policy_features = self.policy.mlp_extractor.policy_net(features)
            else:
                value_features = features
                policy_features = features
            
            value = self.policy.value_net(value_features).cpu().numpy()[0][0]
            
            # Get activations (sample from first hidden layer)
            activations = policy_features.cpu().numpy().flatten()[:50]  # First 50 neurons
            
            return action[0], value, activations
    
    def create_dashboard(self):
        """Create interactive dashboard"""
        # Create figure with subplots
        fig = plt.figure(figsize=(18, 10))
        gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)
        
        # 1. Rocket State (top left)
        ax_state = fig.add_subplot(gs[0, 0])
        ax_state.set_title('Rocket State', fontsize=12, fontweight='bold')
        
        # 2. Actions (top middle)
        ax_actions = fig.add_subplot(gs[0, 1])
        ax_actions.set_title('Network Output (Actions)', fontsize=12, fontweight='bold')
        
        # 3. Value Estimate (top right)
        ax_value = fig.add_subplot(gs[0, 2])
        ax_value.set_title('Value Estimate', fontsize=12, fontweight='bold')
        
        # 4. Neuron Activations (middle row)
        ax_neurons = fig.add_subplot(gs[1, :])
        ax_neurons.set_title('Hidden Layer Neuron Activations (First 50 neurons)', 
                            fontsize=12, fontweight='bold')
        
        # 5. Observation History (bottom left)
        ax_obs_hist = fig.add_subplot(gs[2, 0])
        ax_obs_hist.set_title('Altitude & Velocity', fontsize=10, fontweight='bold')
        
        # 6. Action History (bottom middle)
        ax_action_hist = fig.add_subplot(gs[2, 1])
        ax_action_hist.set_title('Action History', fontsize=10, fontweight='bold')
        
        # 7. Reward History (bottom right)
        ax_reward_hist = fig.add_subplot(gs[2, 2])
        ax_reward_hist.set_title('Reward & Value', fontsize=10, fontweight='bold')
        
        # Animation update function
        def update(frame):
            # Get current data
            action, value, activations = self.get_network_data(self.obs)
            
            # Take step
            next_obs, reward, done, info = self.env.step([action])
            self.episode_reward += reward[0]
            self.step_count += 1
            
            # Store history
            self.obs_history.append(self.obs[0].copy())
            self.action_history.append(action.copy())
            self.reward_history.append(reward[0])
            self.value_history.append(value)
            
            # Limit history
            if len(self.obs_history) > self.max_history:
                self.obs_history.pop(0)
                self.action_history.pop(0)
                self.reward_history.pop(0)
                self.value_history.pop(0)
            
            self.obs = next_obs
            
            if done:
                print(f"Episode finished! Reward: {self.episode_reward:.2f}, Steps: {self.step_count}")
                self.obs = self.env.reset()
                self.episode_reward = 0
                self.step_count = 0
                self.obs_history = []
                self.action_history = []
                self.reward_history = []
                self.value_history = []
            
            # Clear all axes
            for ax in [ax_state, ax_actions, ax_value, ax_neurons, 
                      ax_obs_hist, ax_action_hist, ax_reward_hist]:
                ax.clear()
            
            # =====================================================================
            # 1. ROCKET STATE
            # =====================================================================
            obs_current = self.obs[0]
            state_labels = ['X', 'Y', 'Z', 'Vx', 'Vy', 'Vz', 'Qx', 'Qy', 'Qz', 'Qw', 
                          'ωx', 'ωy', 'ωz', 'Fuel']
            state_colors = ['b']*3 + ['g']*3 + ['orange']*4 + ['purple']*3 + ['r']
            
            bars = ax_state.barh(state_labels, obs_current, color=state_colors, alpha=0.7)
            ax_state.set_xlabel('Value')
            ax_state.set_title('Rocket State', fontsize=12, fontweight='bold')
            ax_state.axvline(x=0, color='black', linestyle='--', linewidth=0.5)
            ax_state.grid(axis='x', alpha=0.3)
            
            # =====================================================================
            # 2. ACTIONS
            # =====================================================================
            action_labels = ['Main', 'North', 'East', 'South', 'West']
            action_colors = ['red', 'blue', 'green', 'orange', 'purple']
            
            bars = ax_actions.bar(action_labels, action, color=action_colors, alpha=0.7)
            ax_actions.set_ylabel('Throttle [0-1]')
            ax_actions.set_ylim(0, 1)
            ax_actions.set_title('Network Output (Actions)', fontsize=12, fontweight='bold')
            ax_actions.grid(axis='y', alpha=0.3)
            
            # Add value labels
            for bar in bars:
                height = bar.get_height()
                ax_actions.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                              f'{height:.2f}', ha='center', va='bottom', fontsize=9)
            
            # =====================================================================
            # 3. VALUE ESTIMATE
            # =====================================================================
            ax_value.bar(['Value'], [value], color='gold', alpha=0.7, width=0.5)
            ax_value.set_ylabel('Expected Return')
            ax_value.axhline(y=0, color='black', linestyle='--', linewidth=1)
            ax_value.text(0, value + 0.5, f'{value:.2f}', ha='center', va='bottom', 
                        fontsize=14, fontweight='bold')
            ax_value.set_title('Value Estimate', fontsize=12, fontweight='bold')
            ax_value.grid(axis='y', alpha=0.3)
            
            # =====================================================================
            # 4. NEURON ACTIVATIONS
            # =====================================================================
            neuron_indices = np.arange(len(activations))
            colors = plt.cm.RdYlGn((activations - activations.min()) / 
                                   (activations.max() - activations.min() + 1e-8))
            
            ax_neurons.bar(neuron_indices, activations, color=colors, alpha=0.8)
            ax_neurons.set_xlabel('Neuron Index')
            ax_neurons.set_ylabel('Activation')
            ax_neurons.set_title('Hidden Layer Neuron Activations', fontsize=12, fontweight='bold')
            ax_neurons.axhline(y=0, color='black', linestyle='--', linewidth=0.5)
            ax_neurons.grid(axis='y', alpha=0.3)
            
            # =====================================================================
            # 5. OBSERVATION HISTORY
            # =====================================================================
            if len(self.obs_history) > 1:
                obs_array = np.array(self.obs_history)
                steps = np.arange(len(self.obs_history))
                
                ax_obs_hist.plot(steps, obs_array[:, 2], label='Altitude (Z)', 
                               color='blue', linewidth=2)
                ax_obs_hist.plot(steps, obs_array[:, 5], label='Vertical Vel (Vz)', 
                               color='green', linewidth=2)
                ax_obs_hist.axhline(y=0, color='black', linestyle='--', linewidth=0.5)
                ax_obs_hist.set_xlabel('Step')
                ax_obs_hist.set_ylabel('Value')
                ax_obs_hist.legend(loc='upper right', fontsize=8)
                ax_obs_hist.grid(alpha=0.3)
            
            # =====================================================================
            # 6. ACTION HISTORY
            # =====================================================================
            if len(self.action_history) > 1:
                action_array = np.array(self.action_history)
                steps = np.arange(len(self.action_history))
                
                ax_action_hist.plot(steps, action_array[:, 0], label='Main Thrust', 
                                   color='red', linewidth=2)
                ax_action_hist.plot(steps, action_array[:, 1:].mean(axis=1), 
                                   label='Avg Thrusters', color='blue', linewidth=2, 
                                   linestyle='--')
                ax_action_hist.set_xlabel('Step')
                ax_action_hist.set_ylabel('Throttle')
                ax_action_hist.set_ylim(0, 1)
                ax_action_hist.legend(loc='upper right', fontsize=8)
                ax_action_hist.grid(alpha=0.3)
            
            # =====================================================================
            # 7. REWARD & VALUE HISTORY
            # =====================================================================
            if len(self.reward_history) > 1:
                steps = np.arange(len(self.reward_history))
                
                ax_reward_hist.plot(steps, self.reward_history, label='Reward', 
                                   color='green', linewidth=2)
                ax_reward_hist.plot(steps, self.value_history, label='Value', 
                                   color='gold', linewidth=2, linestyle='--')
                ax_reward_hist.axhline(y=0, color='black', linestyle='--', linewidth=0.5)
                ax_reward_hist.set_xlabel('Step')
                ax_reward_hist.set_ylabel('Value')
                ax_reward_hist.legend(loc='upper right', fontsize=8)
                ax_reward_hist.grid(alpha=0.3)
            
            # Overall title
            info_text = f"Step: {self.step_count} | Episode Reward: {self.episode_reward:.2f}"
            fig.suptitle(f'Live Neural Network Dashboard - {info_text}', 
                        fontsize=14, fontweight='bold')
        
        # Create animation
        anim = FuncAnimation(fig, update, interval=50, cache_frame_data=False)
        
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    import sys
    
    model_path = "best_model_curriculum/best_model"
    vecnorm_path = "models_curriculum/falcon9_final_vecnorm.pkl"
    
    if not os.path.exists(model_path + ".zip"):
        print(f"❌ Model not found: {model_path}.zip")
        print("\nUsage: python live_neural_dashboard.py [model_path] [vecnorm_path]")
        sys.exit(1)
    
    if len(sys.argv) > 1:
        model_path = sys.argv[1]
    if len(sys.argv) > 2:
        vecnorm_path = sys.argv[2]
    
    print("\n" + "="*70)
    print("🎛️  LIVE NEURAL NETWORK DASHBOARD")
    print("="*70)
    print("\nStarting live visualization...")
    print("Close the window to stop.\n")
    
    dashboard = LiveNeuralDashboard(model_path, vecnorm_path)
    dashboard.create_dashboard()