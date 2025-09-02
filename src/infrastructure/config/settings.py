"""
Configurações gerais do sistema.
"""
from pathlib import Path
from typing import Dict, Any
import os

class Settings:
    """Configurações centralizadas do sistema."""
    
    # Configurações de ML
    ML_CONFIDENCE_THRESHOLD: float = 0.55
    ML_IMAGE_SIZE: int = 736
    ML_AXLE_GROUPING_THRESHOLD: int = 40
    ML_LIFTED_AXLE_THRESHOLD: int = 5
    
    # Configurações de processamento
    PROCESSING_BATCH_SIZE: int = 1
    PROCESSING_TIMEOUT: int = 30
    
    # Configurações de output
    OUTPUT_IMAGE_QUALITY: int = 95
    OUTPUT_JSON_INDENT: int = 2
    
    # Configurações de logging
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # Configurações de categorização
    EXPECTED_AXLES: int = 3
    CONFIDENCE_THRESHOLDS: Dict[str, float] = {
        "low": 0.7,
        "high": 0.8
    }
    
    @classmethod
    def get_ml_config(cls) -> Dict[str, Any]:
        """Retorna configurações de ML."""
        return {
            "confidence_threshold": cls.ML_CONFIDENCE_THRESHOLD,
            "image_size": cls.ML_IMAGE_SIZE,
            "axle_grouping_threshold": cls.ML_AXLE_GROUPING_THRESHOLD,
            "lifted_axle_threshold": cls.ML_LIFTED_AXLE_THRESHOLD
        }
    
    @classmethod
    def get_processing_config(cls) -> Dict[str, Any]:
        """Retorna configurações de processamento."""
        return {
            "batch_size": cls.PROCESSING_BATCH_SIZE,
            "timeout": cls.PROCESSING_TIMEOUT
        }
    
    @classmethod
    def get_output_config(cls) -> Dict[str, Any]:
        """Retorna configurações de output."""
        return {
            "image_quality": cls.OUTPUT_IMAGE_QUALITY,
            "json_indent": cls.OUTPUT_JSON_INDENT
        }

# Instância global das configurações
settings = Settings()
