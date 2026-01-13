"""
Held-out object set for evaluation and generalization testing.

This module manages the separation between training and evaluation objects,
ensuring that evaluation uses objects not seen during training.
"""

import numpy as np
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
from experiments.config import CurriculumConfig


@dataclass
class ObjectProperties:
    """
    Properties defining a unique object configuration.
    
    In our simplified simulation, objects are defined by their
    physical properties rather than geometric shapes.
    """
    size: float
    mass: float
    friction: float
    
    def __hash__(self):
        """Make ObjectProperties hashable for set operations."""
        return hash((round(self.size, 4), round(self.mass, 4), round(self.friction, 4)))
    
    def __eq__(self, other):
        """Compare object properties."""
        if not isinstance(other, ObjectProperties):
            return False
        return (round(self.size, 4) == round(other.size, 4) and
                round(self.mass, 4) == round(other.mass, 4) and
                round(self.friction, 4) == round(other.friction, 4))


class HeldOutObjectSet:
    """
    Manages held-out objects for evaluation.
    
    Ensures strict separation between training and evaluation objects.
    """
    
    def __init__(
        self,
        train_config: CurriculumConfig,
        eval_size_range: Optional[Tuple[float, float]] = None,
        eval_mass_range: Optional[Tuple[float, float]] = None,
        eval_friction_range: Optional[Tuple[float, float]] = None,
        num_heldout_objects: int = 20,
        seed: int = 42
    ):
        """
        Initialize held-out object set.
        
        Args:
            train_config: Training configuration (to avoid overlap)
            eval_size_range: Size range for evaluation objects (if None, uses different range)
            eval_mass_range: Mass range for evaluation objects
            eval_friction_range: Friction range for evaluation objects
            num_heldout_objects: Number of held-out object configurations
            seed: Random seed for reproducibility
        """
        self.train_config = train_config
        self.num_heldout_objects = num_heldout_objects
        self.rng = np.random.default_rng(seed)
        
        # Define evaluation ranges (different from training to ensure separation)
        if eval_size_range is None:
            # Use different range than training
            if train_config.object_size_range:
                train_min, train_max = train_config.object_size_range
                # Shift range to avoid overlap
                range_size = train_max - train_min
                eval_size_range = (train_max + 0.01, train_max + 0.01 + range_size)
            else:
                eval_size_range = (0.06, 0.10)  # Different from default
        
        if eval_mass_range is None:
            if train_config.object_mass_range:
                train_min, train_max = train_config.object_mass_range
                range_size = train_max - train_min
                eval_mass_range = (train_max + 0.01, train_max + 0.01 + range_size)
            else:
                eval_mass_range = (0.15, 0.25)  # Different from default
        
        if eval_friction_range is None:
            if train_config.friction_range:
                train_min, train_max = train_config.friction_range
                range_size = train_max - train_min
                eval_friction_range = (max(0.0, train_min - range_size), train_min - 0.01)
            else:
                eval_friction_range = (0.2, 0.4)  # Different from default
        
        self.eval_size_range = eval_size_range
        self.eval_mass_range = eval_mass_range
        self.eval_friction_range = eval_friction_range
        
        # Generate held-out objects
        self.heldout_objects: List[ObjectProperties] = []
        self._generate_heldout_objects()
    
    def _generate_heldout_objects(self):
        """Generate held-out object configurations."""
        for _ in range(self.num_heldout_objects):
            size = float(self.rng.uniform(self.eval_size_range[0], self.eval_size_range[1]))
            mass = float(self.rng.uniform(self.eval_mass_range[0], self.eval_mass_range[1]))
            friction = float(self.rng.uniform(self.eval_friction_range[0], self.eval_friction_range[1]))
            
            obj = ObjectProperties(size=size, mass=mass, friction=friction)
            self.heldout_objects.append(obj)
    
    def get_eval_config(self, object_idx: Optional[int] = None) -> CurriculumConfig:
        """
        Get curriculum configuration for a held-out object.
        
        Args:
            object_idx: Index of held-out object (if None, random)
            
        Returns:
            CurriculumConfig for evaluation
        """
        if object_idx is None:
            object_idx = self.rng.integers(0, len(self.heldout_objects))
        
        obj = self.heldout_objects[object_idx % len(self.heldout_objects)]
        
        # Create config with fixed properties (no randomization for eval)
        return CurriculumConfig(
            object_size=obj.size,
            object_size_range=None,  # Fixed for evaluation
            object_mass=obj.mass,
            object_mass_range=None,  # Fixed for evaluation
            friction_coefficient=obj.friction,
            friction_range=None,  # Fixed for evaluation
            spawn_distance=self.train_config.spawn_distance,
            spawn_distance_range=self.train_config.spawn_distance_range,
            spawn_x_range=self.train_config.spawn_x_range,
            spawn_y_range=self.train_config.spawn_y_range,
            spawn_z_range=self.train_config.spawn_z_range,
        )
    
    def get_all_eval_configs(self) -> List[CurriculumConfig]:
        """
        Get all held-out object configurations.
        
        Returns:
            List of CurriculumConfig for all held-out objects
        """
        return [self.get_eval_config(i) for i in range(len(self.heldout_objects))]
    
    def verify_separation(self, train_objects: List[ObjectProperties]) -> bool:
        """
        Verify that held-out objects don't overlap with training objects.
        
        Args:
            train_objects: List of training object properties
            
        Returns:
            True if separation is verified
        """
        train_set = set(train_objects)
        heldout_set = set(self.heldout_objects)
        
        overlap = train_set.intersection(heldout_set)
        return len(overlap) == 0
    
    def get_statistics(self) -> Dict:
        """
        Get statistics about held-out object set.
        
        Returns:
            Dictionary with statistics
        """
        sizes = [obj.size for obj in self.heldout_objects]
        masses = [obj.mass for obj in self.heldout_objects]
        frictions = [obj.friction for obj in self.heldout_objects]
        
        return {
            "num_objects": len(self.heldout_objects),
            "size_range": (float(np.min(sizes)), float(np.max(sizes))),
            "mass_range": (float(np.min(masses)), float(np.max(masses))),
            "friction_range": (float(np.min(frictions)), float(np.max(frictions))),
            "mean_size": float(np.mean(sizes)),
            "mean_mass": float(np.mean(masses)),
            "mean_friction": float(np.mean(frictions)),
        }


def generate_training_objects(
    config: CurriculumConfig,
    num_samples: int = 100,
    seed: int = 42
) -> List[ObjectProperties]:
    """
    Generate training object properties from a curriculum configuration.
    
    Args:
        config: Curriculum configuration
        num_samples: Number of samples to generate
        seed: Random seed
        
    Returns:
        List of ObjectProperties seen during training
    """
    rng = np.random.default_rng(seed)
    objects = []
    
    for _ in range(num_samples):
        size = config.get_object_size(rng)
        mass = config.get_object_mass(rng)
        friction = config.get_friction_coefficient(rng)
        
        obj = ObjectProperties(size=size, mass=mass, friction=friction)
        objects.append(obj)
    
    return objects
