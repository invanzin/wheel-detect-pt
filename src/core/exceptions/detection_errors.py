"""
Exceções customizadas para o sistema de detecção.
"""

class DetectionError(Exception):
    """Exceção base para erros de detecção."""
    pass

class ModelLoadError(DetectionError):
    """Erro ao carregar modelo de ML."""
    pass

class ImageLoadError(DetectionError):
    """Erro ao carregar imagem."""
    pass

class ValidationError(DetectionError):
    """Erro de validação de dados."""
    pass

class ProcessingError(DetectionError):
    """Erro durante o processamento."""
    pass

class OutputError(DetectionError):
    """Erro ao salvar output."""
    pass

class ConfigurationError(DetectionError):
    """Erro de configuração."""
    pass
