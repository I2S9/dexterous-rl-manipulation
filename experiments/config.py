"""
Configuration system for curriculum learning parameters.

This module defines the curriculum variables that control task difficulty:
- Object size
- Object mass
- Friction coefficient
- Spawn distance (distance from hand to object)
"""

from dataclasses import dataclass
from typing import Optional, Tuple
import json
from pathlib import Path


@dataclass
class CurriculumConfig:
    """
    Configuration for curriculum learning parameters.
    
    These parameters control the difficulty of the manipulation task.
    """
    
    # Object properties
    object_size: float = 0.05  # Object radius in meters (default: 5cm)
    object_size_range: Optional[Tuple[float, float]] = None  # (min, max) for randomization
    
    object_mass: float = 0.1  # Object mass in kg (default: 100g)
    object_mass_range: Optional[Tuple[float, float]] = None  # (min, max) for randomization
    
    friction_coefficient: float = 0.5  # Friction coefficient (default: 0.5)
    friction_range: Optional[Tuple[float, float]] = None  # (min, max) for randomization
    
    # Spawn configuration
    spawn_distance: float = 0.15  # Distance from hand base to object in meters (default: 15cm)
    spawn_distance_range: Optional[Tuple[float, float]] = None  # (min, max) for randomization
    
    # Spawn position bounds (relative to hand)
    spawn_x_range: Tuple[float, float] = (-0.1, 0.1)  # X position range
    spawn_y_range: Tuple[float, float] = (-0.1, 0.1)  # Y position range
    spawn_z_range: Tuple[float, float] = (0.05, 0.2)  # Z position range
    
    def get_object_size(self, rng) -> float:
        """
        Get object size, potentially randomized within range.
        
        Args:
            rng: Random number generator
            
        Returns:
            Object size in meters
        """
        if self.object_size_range is not None:
            return float(rng.uniform(self.object_size_range[0], self.object_size_range[1]))
        return self.object_size
    
    def get_object_mass(self, rng) -> float:
        """
        Get object mass, potentially randomized within range.
        
        Args:
            rng: Random number generator
            
        Returns:
            Object mass in kg
        """
        if self.object_mass_range is not None:
            return float(rng.uniform(self.object_mass_range[0], self.object_mass_range[1]))
        return self.object_mass
    
    def get_friction_coefficient(self, rng) -> float:
        """
        Get friction coefficient, potentially randomized within range.
        
        Args:
            rng: Random number generator
            
        Returns:
            Friction coefficient
        """
        if self.friction_range is not None:
            return float(rng.uniform(self.friction_range[0], self.friction_range[1]))
        return self.friction_coefficient
    
    def get_spawn_distance(self, rng) -> float:
        """
        Get spawn distance, potentially randomized within range.
        
        Args:
            rng: Random number generator
            
        Returns:
            Spawn distance in meters
        """
        if self.spawn_distance_range is not None:
            return float(rng.uniform(self.spawn_distance_range[0], self.spawn_distance_range[1]))
        return self.spawn_distance
    
    def get_spawn_position(self, rng) -> Tuple[float, float, float]:
        """
        Get randomized spawn position within configured ranges.
        
        Args:
            rng: Random number generator
            
        Returns:
            (x, y, z) spawn position in meters
        """
        x = float(rng.uniform(self.spawn_x_range[0], self.spawn_x_range[1]))
        y = float(rng.uniform(self.spawn_y_range[0], self.spawn_y_range[1]))
        z = float(rng.uniform(self.spawn_z_range[0], self.spawn_z_range[1]))
        return (x, y, z)
    
    @classmethod
    def from_dict(cls, config_dict: dict) -> "CurriculumConfig":
        """
        Create CurriculumConfig from dictionary.
        
        Args:
            config_dict: Dictionary with configuration parameters
            
        Returns:
            CurriculumConfig instance
        """
        return cls(**config_dict)
    
    def to_dict(self) -> dict:
        """
        Convert configuration to dictionary.
        
        Returns:
            Dictionary representation of configuration
        """
        return {
            "object_size": self.object_size,
            "object_size_range": self.object_size_range,
            "object_mass": self.object_mass,
            "object_mass_range": self.object_mass_range,
            "friction_coefficient": self.friction_coefficient,
            "friction_range": self.friction_range,
            "spawn_distance": self.spawn_distance,
            "spawn_distance_range": self.spawn_distance_range,
            "spawn_x_range": self.spawn_x_range,
            "spawn_y_range": self.spawn_y_range,
            "spawn_z_range": self.spawn_z_range,
        }
    
    @classmethod
    def from_json(cls, json_path: str) -> "CurriculumConfig":
        """
        Load configuration from JSON file.
        
        Args:
            json_path: Path to JSON configuration file
            
        Returns:
            CurriculumConfig instance
        """
        with open(json_path, 'r') as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)
    
    def to_json(self, json_path: str):
        """
        Save configuration to JSON file.
        
        Args:
            json_path: Path to save JSON configuration file
        """
        with open(json_path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def easy(cls) -> "CurriculumConfig":
        """
        Create easy curriculum configuration.
        
        Returns:
            Easy difficulty configuration
        """
        return cls(
            object_size=0.08,  # Larger object (easier to grasp)
            object_mass=0.05,  # Lighter object (easier to manipulate)
            friction_coefficient=0.8,  # Higher friction (easier to hold)
            spawn_distance=0.10,  # Closer to hand (easier to reach)
        )
    
    @classmethod
    def medium(cls) -> "CurriculumConfig":
        """
        Create medium curriculum configuration.
        
        Returns:
            Medium difficulty configuration
        """
        return cls(
            object_size=0.05,
            object_mass=0.1,
            friction_coefficient=0.5,
            spawn_distance=0.15,
        )
    
    @classmethod
    def hard(cls) -> "CurriculumConfig":
        """
        Create hard curriculum configuration.
        
        Returns:
            Hard difficulty configuration
        """
        return cls(
            object_size=0.03,  # Smaller object (harder to grasp)
            object_mass=0.2,  # Heavier object (harder to manipulate)
            friction_coefficient=0.3,  # Lower friction (harder to hold)
            spawn_distance=0.20,  # Farther from hand (harder to reach)
        )
