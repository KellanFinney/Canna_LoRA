"""
Configuration loader for LoRA DataGen pipeline
"""
import yaml
import os
from pathlib import Path
from typing import Dict, Any

class Config:
    """Configuration manager for LoRA DataGen"""
    
    def __init__(self, config_path: str = "config.yaml"):
        self.config_path = Path(config_path)
        self._config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        
        with open(self.config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        # Expand environment variables and relative paths
        config = self._expand_paths(config)
        return config
    
    def _expand_paths(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Expand environment variables and relative paths"""
        for section, settings in config.items():
            if isinstance(settings, dict):
                for key, value in settings.items():
                    if isinstance(value, str):
                        # Expand environment variables
                        value = os.path.expandvars(value)
                        # Convert relative paths to absolute
                        if 'directory' in key.lower() or 'file' in key.lower():
                            if not os.path.isabs(value):
                                value = str(Path(value).resolve())
                        settings[key] = value
        return config
    
    def get(self, section: str, key: str = None, default=None):
        """Get configuration value"""
        try:
            if key is None:
                return self._config.get(section, default)
            return self._config.get(section, {}).get(key, default)
        except KeyError:
            return default
    
    @property
    def data_generation(self) -> Dict[str, Any]:
        """Data generation settings"""
        return self.get('data_generation', default={})
    
    @property
    def openai(self) -> Dict[str, Any]:
        """OpenAI API settings"""
        return self.get('openai', default={})
    
    @property
    def processing(self) -> Dict[str, Any]:
        """Processing settings"""
        return self.get('processing', default={})
    
    @property
    def quality(self) -> Dict[str, Any]:
        """Quality assessment settings"""
        return self.get('quality', default={})
    
    @property
    def data_loading(self) -> Dict[str, Any]:
        """Data loading settings"""
        return self.get('data_loading', default={})
    
    @property
    def huggingface(self) -> Dict[str, Any]:
        """HuggingFace integration settings"""
        return self.get('huggingface', default={})

# Global config instance
_config = None

def get_config(config_path: str = "config.yaml") -> Config:
    """Get global configuration instance"""
    global _config
    if _config is None:
        _config = Config(config_path)
    return _config

def reload_config(config_path: str = "config.yaml") -> Config:
    """Reload configuration from file"""
    global _config
    _config = Config(config_path)
    return _config 