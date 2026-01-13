"""
Dexterous manipulation environment for reinforcement learning.

This module implements a Gymnasium-compatible environment for training
policies on dexterous robotic hand manipulation tasks.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, Tuple, Optional, Any


class DexterousManipulationEnv(gym.Env):
    """
    Environment for dexterous manipulation with a multi-fingered robotic hand.
    
    The environment simulates a robotic hand attempting to grasp and manipulate
    objects. The hand has multiple fingers with continuous joint control.
    """
    
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}
    
    def __init__(
        self,
        num_fingers: int = 5,
        joints_per_finger: int = 3,
        object_position: Optional[np.ndarray] = None,
        max_episode_steps: int = 200,
        render_mode: Optional[str] = None,
        reward_type: str = "sparse",
        reward_shaping: Optional[Any] = None,
        curriculum_config: Optional[Any] = None,
    ):
        """
        Initialize the dexterous manipulation environment.
        
        Args:
            num_fingers: Number of fingers on the robotic hand (default: 5)
            joints_per_finger: Number of joints per finger (default: 3)
            object_position: Initial object position [x, y, z]. If None, random.
            max_episode_steps: Maximum steps per episode (default: 200)
            render_mode: Rendering mode, one of [None, "human", "rgb_array"]
            reward_type: Type of reward ("sparse" or "dense")
            reward_shaping: Optional reward shaping object (if None, created based on reward_type)
            curriculum_config: Optional curriculum configuration (if None, uses defaults)
        """
        super().__init__()
        
        self.num_fingers = num_fingers
        self.joints_per_finger = joints_per_finger
        self.num_joints = num_fingers * joints_per_finger
        self.max_episode_steps = max_episode_steps
        self.render_mode = render_mode
        self.reward_type = reward_type
        
        # Initialize curriculum configuration
        if curriculum_config is None:
            from experiments import CurriculumConfig
            self.curriculum_config = CurriculumConfig()
        else:
            self.curriculum_config = curriculum_config
        
        # Initialize reward shaping
        if reward_shaping is not None:
            self.reward_shaping = reward_shaping
        else:
            if reward_type == "dense":
                from rewards import RewardShaping
                self.reward_shaping = RewardShaping()
            else:
                from rewards import SparseReward
                self.reward_shaping = SparseReward()
        
        # Store finger tips for reward computation
        self.finger_tips = None
        
        # Curriculum variables (set during reset)
        self.object_size = None
        self.object_mass = None
        self.friction_coefficient = None
        
        # Action space: continuous control for each joint
        # Actions are in range [-1, 1], representing normalized joint velocities
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.num_joints,),
            dtype=np.float32
        )
        
        # Observation space: hand state + object state
        # Hand state: joint positions (num_joints) + joint velocities (num_joints)
        # Object state: position (3) + orientation quaternion (4) + velocity (3)
        # Contact information: binary contacts per finger (num_fingers)
        hand_state_dim = 2 * self.num_joints  # positions + velocities
        object_state_dim = 3 + 4 + 3  # pos + quat + vel
        contact_dim = self.num_fingers
        
        obs_dim = hand_state_dim + object_state_dim + contact_dim
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32
        )
        
        # Internal state
        self.joint_positions = None
        self.joint_velocities = None
        self.object_position = object_position
        self.object_orientation = None
        self.object_velocity = None
        self.contacts = None
        self.step_count = 0
        self._last_reward_components = None
        
        # Workspace bounds (in meters)
        self.workspace_bounds = np.array([[-0.2, 0.2], [-0.2, 0.2], [0.0, 0.3]])
        
        # Hand base position (center of workspace)
        self.hand_base_position = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict] = None
    ) -> Tuple[np.ndarray, Dict]:
        """
        Reset the environment to an initial state.
        
        Args:
            seed: Random seed for reproducibility
            options: Additional options for reset
            
        Returns:
            observation: Initial observation
            info: Additional information dictionary
        """
        super().reset(seed=seed)
        
        # Reset joint positions to neutral (slightly open hand)
        self.joint_positions = self.np_random.uniform(
            low=-0.1, high=0.1, size=(self.num_joints,)
        ).astype(np.float32)
        
        # Reset joint velocities to zero
        self.joint_velocities = np.zeros(self.num_joints, dtype=np.float32)
        
        # Sample curriculum variables
        self.object_size = self.curriculum_config.get_object_size(self.np_random)
        self.object_mass = self.curriculum_config.get_object_mass(self.np_random)
        self.friction_coefficient = self.curriculum_config.get_friction_coefficient(self.np_random)
        
        # Reset object position using curriculum spawn configuration
        if self.object_position is None:
            # Use curriculum spawn position
            spawn_pos = self.curriculum_config.get_spawn_position(self.np_random)
            self.object_position = np.array(spawn_pos, dtype=np.float32)
        else:
            self.object_position = np.array(self.object_position, dtype=np.float32)
        
        # Reset object orientation (identity quaternion)
        self.object_orientation = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        
        # Reset object velocity
        self.object_velocity = np.zeros(3, dtype=np.float32)
        
        # Reset contacts (no initial contacts)
        self.contacts = np.zeros(self.num_fingers, dtype=np.float32)
        
        # Reset step counter
        self.step_count = 0
        
        # Update finger tips and reset reward shaping
        self._update_contacts()
        self.reward_shaping.reset()
        
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, info
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute one step in the environment.
        
        Args:
            action: Action vector of shape (num_joints,)
            
        Returns:
            observation: Next observation
            reward: Reward for this step
            terminated: Whether episode terminated (task completed)
            truncated: Whether episode truncated (max steps reached)
            info: Additional information dictionary
        """
        # Clip action to valid range
        action = np.clip(action, self.action_space.low, self.action_space.high)
        
        # Update joint positions (simple dynamics: velocity = action)
        dt = 0.01  # 10ms timestep
        self.joint_velocities = 0.9 * self.joint_velocities + 0.1 * action
        self.joint_positions += self.joint_velocities * dt
        
        # Clip joint positions to reasonable range
        self.joint_positions = np.clip(self.joint_positions, -1.0, 1.0)
        
        # Object dynamics (affected by mass and friction)
        # Gravity force depends on mass
        gravity_accel = 9.81  # m/s^2
        gravity = np.array([0.0, 0.0, -gravity_accel * dt])
        
        # Apply friction damping (proportional to friction coefficient)
        friction_damping = 1.0 - (self.friction_coefficient * 0.1 * dt)
        self.object_velocity *= friction_damping
        
        # Apply gravity
        self.object_velocity += gravity
        
        # Update position
        self.object_position += self.object_velocity * dt
        
        # Constrain object to workspace bounds
        self.object_position = np.clip(
            self.object_position,
            self.workspace_bounds[:, 0],
            self.workspace_bounds[:, 1]
        )
        
        # Reset velocity if hitting bounds
        for i in range(3):
            if (self.object_position[i] <= self.workspace_bounds[i, 0] and self.object_velocity[i] < 0) or \
               (self.object_position[i] >= self.workspace_bounds[i, 1] and self.object_velocity[i] > 0):
                self.object_velocity[i] = 0.0
        
        # Check contacts (simplified: contact if fingers are close to object)
        self._update_contacts()
        
        # Compute reward using reward shaping
        reward = self._compute_reward()
        
        # Check termination conditions
        terminated = self._check_termination()
        truncated = self.step_count >= self.max_episode_steps
        
        self.step_count += 1
        
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, reward, terminated, truncated, info
    
    def _get_observation(self) -> np.ndarray:
        """Construct the observation vector."""
        obs_parts = [
            self.joint_positions,
            self.joint_velocities,
            self.object_position,
            self.object_orientation,
            self.object_velocity,
            self.contacts,
        ]
        return np.concatenate(obs_parts, dtype=np.float32)
    
    def _get_info(self) -> Dict[str, Any]:
        """Get additional information about the current state."""
        info = {
            "step_count": self.step_count,
            "object_position": self.object_position.copy(),
            "num_contacts": int(np.sum(self.contacts > 0.5)),
            "curriculum": {
                "object_size": float(self.object_size),
                "object_mass": float(self.object_mass),
                "friction_coefficient": float(self.friction_coefficient),
            },
        }
        
        # Add reward components if available
        if self._last_reward_components is not None:
            info["reward_components"] = self._last_reward_components.copy()
        
        return info
    
    def _update_contacts(self):
        """
        Update contact information between fingers and object.
        
        Simplified contact model: contact exists if finger tip is within
        a threshold distance of the object. Threshold depends on object size.
        """
        # Contact threshold scales with object size
        contact_threshold = self.object_size * 1.5  # 1.5x object radius
        
        # Simplified finger tip positions (based on joint positions)
        # In a real simulation, this would use forward kinematics
        finger_tips = []
        for i in range(self.num_fingers):
            # Approximate finger tip position
            finger_base = np.array([0.0, 0.0, 0.0])  # Hand base position
            finger_direction = self.joint_positions[i * self.joints_per_finger:(i + 1) * self.joints_per_finger]
            finger_tip = finger_base + np.sum(finger_direction) * 0.1  # Simplified
            finger_tips.append(finger_tip)
        
        finger_tips = np.array(finger_tips)
        self.finger_tips = finger_tips
        
        # Check distance to object
        distances = np.linalg.norm(finger_tips - self.object_position, axis=1)
        self.contacts = (distances < contact_threshold).astype(np.float32)
    
    def _compute_reward(self) -> float:
        """
        Compute reward for current state using reward shaping.
        
        Uses either sparse or dense reward formulation based on configuration.
        """
        reward_dict = self.reward_shaping.compute(
            joint_positions=self.joint_positions,
            finger_tips=self.finger_tips,
            object_position=self.object_position,
            contacts=self.contacts,
            num_fingers=self.num_fingers,
            joints_per_finger=self.joints_per_finger,
        )
        
        # Store reward components for info dict
        self._last_reward_components = reward_dict
        
        return reward_dict["total"]
    
    def _check_termination(self) -> bool:
        """Check if episode should terminate (task completed)."""
        # Episode terminates if object is successfully grasped
        num_contacts = np.sum(self.contacts > 0.5)
        return num_contacts >= 3
    
    def render(self):
        """Render the environment (placeholder for visualization)."""
        if self.render_mode == "human":
            # Placeholder for future visualization
            pass
        elif self.render_mode == "rgb_array":
            # Placeholder for future image rendering
            return np.zeros((480, 640, 3), dtype=np.uint8)
    
    def close(self):
        """Clean up resources."""
        pass
