"""
Policy pool for storing historical policies in PSRO.
"""
import os
import json
from typing import Optional, Dict, Any

class PolicyPool:
    """
    Manages a pool of trained policies for PSRO.
    """
    
    def __init__(self, save_dir: str):
        """
        Args:
            save_dir: Directory to save policy checkpoints
        """
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
        # Load existing policies
        self.policies = []
        self._load_manifest()
    
    def _load_manifest(self):
        """Load manifest file listing all policies."""
        manifest_path = os.path.join(self.save_dir, 'manifest.json')
        if os.path.exists(manifest_path):
            with open(manifest_path, 'r') as f:
                self.policies = json.load(f)
    
    def _save_manifest(self):
        """Save manifest file."""
        manifest_path = os.path.join(self.save_dir, 'manifest.json')
        with open(manifest_path, 'w') as f:
            json.dump(self.policies, f, indent=2)
    
    def add(self, agent, metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Add a trained agent to the pool.
        
        Args:
            agent: Agent object with save() method
            metadata: Optional metadata dict
            
        Returns:
            Path to saved checkpoint
        """
        policy_id = len(self.policies)
        checkpoint_name = f'policy_{policy_id}.pt'
        checkpoint_path = os.path.join(self.save_dir, checkpoint_name)
        
        # Save agent
        agent.save(checkpoint_path)
        
        # Add to manifest
        policy_info = {
            'id': policy_id,
            'checkpoint': checkpoint_name,
            'metadata': metadata or {}
        }
        self.policies.append(policy_info)
        self._save_manifest()
        
        return checkpoint_path
    
    def sample_latest(self) -> str:
        """
        Get path to most recent policy.
        
        Returns:
            Path to checkpoint file
        """
        if len(self.policies) == 0:
            raise ValueError("Policy pool is empty")
        
        latest = self.policies[-1]
        return os.path.join(self.save_dir, latest['checkpoint'])
    
    def sample_random(self) -> str:
        """
        Get path to random policy from pool.
        
        Returns:
            Path to checkpoint file
        """
        if len(self.policies) == 0:
            raise ValueError("Policy pool is empty")
        
        import random
        policy = random.choice(self.policies)
        return os.path.join(self.save_dir, policy['checkpoint'])
    
    def size(self) -> int:
        """Return number of policies in pool."""
        return len(self.policies)
    
    def get_metadata(self, policy_id: int) -> Dict[str, Any]:
        """Get metadata for a specific policy."""
        if policy_id >= len(self.policies):
            raise ValueError(f"Policy {policy_id} does not exist")
        return self.policies[policy_id]['metadata']