"""
Feature flag system for gradual rollouts and A/B testing in Yappy.
Allows enabling/disabling features for specific users or percentages.
"""

import json
import hashlib
from typing import Dict, Any, Optional
from pathlib import Path

class FeatureFlags:
    def __init__(self, config_path: str = "feature_flags.json"):
        self.config_path = Path(config_path)
        self.flags = self._load_flags()
    
    def _load_flags(self) -> Dict[str, Any]:
        """Load feature flags from config file."""
        if self.config_path.exists():
            try:
                with open(self.config_path, 'r') as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                pass
        return self._get_default_flags()
    
    def _get_default_flags(self) -> Dict[str, Any]:
        """Return default feature flag configuration."""
        return {
            "voice_input": {
                "enabled": False,
                "rollout_percentage": 0,
                "description": "Voice input functionality"
            },
            "dark_theme": {
                "enabled": False,
                "rollout_percentage": 0,
                "description": "Dark theme toggle"
            },
            "conversation_history": {
                "enabled": True,
                "rollout_percentage": 100,
                "description": "Chat conversation history"
            },
            "file_upload": {
                "enabled": False,
                "rollout_percentage": 0,
                "description": "File upload capabilities"
            },
            "advanced_animations": {
                "enabled": True,
                "rollout_percentage": 100,
                "description": "Enhanced Yappy character animations"
            },
            "beta_features": {
                "enabled": False,
                "rollout_percentage": 0,
                "description": "Experimental beta features",
                "allowed_users": []
            }
        }
    
    def is_enabled(self, flag_name: str, user_id: Optional[str] = None) -> bool:
        """Check if a feature flag is enabled for a user."""
        if flag_name not in self.flags:
            return False
        
        flag_config = self.flags[flag_name]
        
        # Check if feature is globally disabled
        if not flag_config.get("enabled", False):
            return False
        
        # Check if user is in allowed users list
        allowed_users = flag_config.get("allowed_users", [])
        if allowed_users and user_id in allowed_users:
            return True
        
        # Check rollout percentage
        rollout_percentage = flag_config.get("rollout_percentage", 0)
        if rollout_percentage >= 100:
            return True
        
        if rollout_percentage <= 0:
            return False
        
        # Use user_id hash for consistent rollout
        if user_id:
            user_hash = int(hashlib.md5(f"{flag_name}:{user_id}".encode()).hexdigest()[:8], 16)
            return (user_hash % 100) < rollout_percentage
        
        return False
    
    def get_enabled_flags(self, user_id: Optional[str] = None) -> Dict[str, bool]:
        """Get all enabled flags for a user."""
        return {
            flag_name: self.is_enabled(flag_name, user_id)
            for flag_name in self.flags.keys()
        }
    
    def update_flag(self, flag_name: str, enabled: bool = None, 
                   rollout_percentage: int = None, allowed_users: list = None):
        """Update a feature flag configuration."""
        if flag_name not in self.flags:
            self.flags[flag_name] = {"enabled": False, "rollout_percentage": 0}
        
        if enabled is not None:
            self.flags[flag_name]["enabled"] = enabled
        if rollout_percentage is not None:
            self.flags[flag_name]["rollout_percentage"] = max(0, min(100, rollout_percentage))
        if allowed_users is not None:
            self.flags[flag_name]["allowed_users"] = allowed_users
        
        self._save_flags()
    
    def _save_flags(self):
        """Save feature flags to config file."""
        try:
            with open(self.config_path, 'w') as f:
                json.dump(self.flags, f, indent=2)
        except IOError:
            pass

# Global instance
feature_flags = FeatureFlags()