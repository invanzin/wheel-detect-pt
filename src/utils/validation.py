"""
Utilitários de validação para dados e parâmetros.
"""
from pathlib import Path
from typing import Any, List, Dict, Optional, Union, Tuple
import numpy as np
import logging

logger = logging.getLogger(__name__)

class ValidationError(Exception):
    """Exceção para erros de validação."""
    pass

class Validator:
    """Classe para validação de dados e parâmetros."""
    
    @staticmethod
    def validate_file_path(file_path: Union[str, Path], 
                          must_exist: bool = True,
                          file_types: Optional[List[str]] = None) -> Path:
        """
        Valida um caminho de arquivo.
        
        Args:
            file_path: Caminho do arquivo
            must_exist: Se o arquivo deve existir
            file_types: Lista de extensões permitidas
            
        Returns:
            Path validado
            
        Raises:
            ValidationError: Se a validação falhar
        """
        try:
            path = Path(file_path)
            
            # Verificar se deve existir
            if must_exist and not path.exists():
                raise ValidationError(f"Arquivo não encontrado: {path}")
            
            # Verificar extensão se especificada
            if file_types and path.suffix.lower() not in [f".{ext.lower()}" for ext in file_types]:
                raise ValidationError(f"Tipo de arquivo não permitido: {path.suffix}. Permitidos: {file_types}")
            
            return path
            
        except Exception as e:
            if isinstance(e, ValidationError):
                raise
            raise ValidationError(f"Erro ao validar caminho {file_path}: {e}")
    
    @staticmethod
    def validate_directory_path(dir_path: Union[str, Path], 
                              must_exist: bool = False,
                              create_if_missing: bool = True) -> Path:
        """
        Valida um caminho de diretório.
        
        Args:
            dir_path: Caminho do diretório
            must_exist: Se o diretório deve existir
            create_if_missing: Se deve criar o diretório se não existir
            
        Returns:
            Path validado
            
        Raises:
            ValidationError: Se a validação falhar
        """
        try:
            path = Path(dir_path)
            
            # Verificar se deve existir
            if must_exist and not path.exists():
                raise ValidationError(f"Diretório não encontrado: {path}")
            
            # Criar se não existir e for permitido
            if create_if_missing and not path.exists():
                path.mkdir(parents=True, exist_ok=True)
                logger.info(f"Diretório criado: {path}")
            
            return path
            
        except Exception as e:
            if isinstance(e, ValidationError):
                raise
            raise ValidationError(f"Erro ao validar diretório {dir_path}: {e}")
    
    @staticmethod
    def validate_numeric_range(value: Union[int, float], 
                              min_value: Optional[Union[int, float]] = None,
                              max_value: Optional[Union[int, float]] = None,
                              name: str = "valor") -> Union[int, float]:
        """
        Valida se um valor numérico está em um intervalo.
        
        Args:
            value: Valor a ser validado
            min_value: Valor mínimo permitido
            max_value: Valor máximo permitido
            name: Nome do valor para mensagens de erro
            
        Returns:
            Valor validado
            
        Raises:
            ValidationError: Se a validação falhar
        """
        if not isinstance(value, (int, float)):
            raise ValidationError(f"{name} deve ser numérico, recebido: {type(value)}")
        
        if min_value is not None and value < min_value:
            raise ValidationError(f"{name} deve ser >= {min_value}, recebido: {value}")
        
        if max_value is not None and value > max_value:
            raise ValidationError(f"{name} deve ser <= {max_value}, recebido: {value}")
        
        return value
    
    @staticmethod
    def validate_list_not_empty(items: List[Any], 
                               name: str = "lista") -> List[Any]:
        """
        Valida se uma lista não está vazia.
        
        Args:
            items: Lista a ser validada
            name: Nome da lista para mensagens de erro
            
        Returns:
            Lista validada
            
        Raises:
            ValidationError: Se a validação falhar
        """
        if not isinstance(items, list):
            raise ValidationError(f"{name} deve ser uma lista, recebido: {type(items)}")
        
        if len(items) == 0:
            raise ValidationError(f"{name} não pode estar vazia")
        
        return items
    
    @staticmethod
    def validate_image_array(image: np.ndarray, 
                           expected_channels: Optional[int] = None,
                           name: str = "imagem") -> np.ndarray:
        """
        Valida um array de imagem.
        
        Args:
            image: Array da imagem
            expected_channels: Número esperado de canais
            name: Nome da imagem para mensagens de erro
            
        Returns:
            Array validado
            
        Raises:
            ValidationError: Se a validação falhar
        """
        if not isinstance(image, np.ndarray):
            raise ValidationError(f"{name} deve ser um numpy array, recebido: {type(image)}")
        
        if image.ndim < 2:
            raise ValidationError(f"{name} deve ter pelo menos 2 dimensões, recebido: {image.ndim}")
        
        if expected_channels is not None:
            actual_channels = image.shape[2] if image.ndim > 2 else 1
            if actual_channels != expected_channels:
                raise ValidationError(f"{name} deve ter {expected_channels} canais, recebido: {actual_channels}")
        
        return image
    
    @staticmethod
    def validate_confidence_threshold(threshold: float, name: str = "threshold") -> float:
        """
        Valida um threshold de confiança.
        
        Args:
            threshold: Threshold a ser validado
            name: Nome do threshold para mensagens de erro
            
        Returns:
            Threshold validado
            
        Raises:
            ValidationError: Se a validação falhar
        """
        return Validator.validate_numeric_range(
            threshold, 
            min_value=0.0, 
            max_value=1.0, 
            name=name
        )
    
    @staticmethod
    def validate_model_config(config: Dict[str, Any], 
                             required_keys: List[str]) -> Dict[str, Any]:
        """
        Valida uma configuração de modelo.
        
        Args:
            config: Configuração a ser validada
            required_keys: Chaves obrigatórias
            
        Returns:
            Configuração validada
            
        Raises:
            ValidationError: Se a validação falhar
        """
        if not isinstance(config, dict):
            raise ValidationError(f"Configuração deve ser um dicionário, recebido: {type(config)}")
        
        missing_keys = [key for key in required_keys if key not in config]
        if missing_keys:
            raise ValidationError(f"Chaves obrigatórias ausentes: {missing_keys}")
        
        return config
    
    @staticmethod
    def validate_coordinates(coordinates: List[Tuple[float, float]], 
                           min_points: int = 1,
                           name: str = "coordenadas") -> List[Tuple[float, float]]:
        """
        Valida uma lista de coordenadas.
        
        Args:
            coordinates: Lista de coordenadas
            min_points: Número mínimo de pontos
            name: Nome das coordenadas para mensagens de erro
            
        Returns:
            Lista de coordenadas validada
            
        Raises:
            ValidationError: Se a validação falhar
        """
        coordinates = Validator.validate_list_not_empty(coordinates, name)
        
        if len(coordinates) < min_points:
            raise ValidationError(f"{name} deve ter pelo menos {min_points} pontos, recebido: {len(coordinates)}")
        
        for i, coord in enumerate(coordinates):
            if not isinstance(coord, tuple) or len(coord) != 2:
                raise ValidationError(f"Coordenada {i} deve ser uma tupla de 2 elementos, recebido: {coord}")
            
            x, y = coord
            if not isinstance(x, (int, float)) or not isinstance(y, (int, float)):
                raise ValidationError(f"Coordenada {i} deve conter valores numéricos, recebido: {coord}")
        
        return coordinates
