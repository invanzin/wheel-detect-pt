"""
Utilitários para manipulação de imagens.
"""
import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, List
import logging

logger = logging.getLogger(__name__)

class ImageUtils:
    """Utilitários para manipulação de imagens."""
    
    @staticmethod
    def load_image(image_path: Path) -> Optional[np.ndarray]:
        """Carrega uma imagem do disco."""
        try:
            if not image_path.exists():
                logger.error(f"Arquivo de imagem não encontrado: {image_path}")
                return None
            
            image = cv2.imread(str(image_path))
            if image is None:
                logger.error(f"Falha ao carregar imagem: {image_path}")
                return None
            
            logger.info(f"Imagem carregada com sucesso: {image_path}")
            return image
            
        except Exception as e:
            logger.error(f"Erro ao carregar imagem {image_path}: {e}")
            return None
    
    @staticmethod
    def save_image(image: np.ndarray, output_path: Path, quality: int = 95) -> bool:
        """Salva uma imagem no disco."""
        try:
            # Criar diretório se não existir
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Salvar imagem
            success = cv2.imwrite(str(output_path), image, [cv2.IMWRITE_JPEG_QUALITY, quality])
            
            if success:
                logger.info(f"Imagem salva com sucesso: {output_path}")
                return True
            else:
                logger.error(f"Falha ao salvar imagem: {output_path}")
                return False
                
        except Exception as e:
            logger.error(f"Erro ao salvar imagem {output_path}: {e}")
            return False
    
    @staticmethod
    def resize_image(image: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
        """Redimensiona uma imagem para o tamanho especificado."""
        return cv2.resize(image, target_size)
    
    @staticmethod
    def draw_bounding_box(image: np.ndarray, bbox: Tuple[int, int, int, int], 
                         color: Tuple[int, int, int] = (0, 255, 0), 
                         thickness: int = 2) -> np.ndarray:
        """Desenha um bounding box na imagem."""
        x1, y1, x2, y2 = bbox
        cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
        return image
    
    @staticmethod
    def draw_text(image: np.ndarray, text: str, position: Tuple[int, int], 
                  font_scale: float = 1.0, color: Tuple[int, int, int] = (255, 255, 255),
                  thickness: int = 2) -> np.ndarray:
        """Desenha texto na imagem."""
        cv2.putText(image, text, position, cv2.FONT_HERSHEY_SIMPLEX, 
                   font_scale, color, thickness)
        return image
    
    @staticmethod
    def get_image_info(image: np.ndarray) -> dict:
        """Retorna informações básicas da imagem."""
        height, width = image.shape[:2]
        channels = image.shape[2] if len(image.shape) > 2 else 1
        
        return {
            "height": height,
            "width": width,
            "channels": channels,
            "dtype": str(image.dtype),
            "size_bytes": image.nbytes
        }
    
    @staticmethod
    def convert_color_space(image: np.ndarray, from_space: str, to_space: str) -> np.ndarray:
        """Converte imagem entre espaços de cor."""
        color_conversions = {
            ("BGR", "RGB"): cv2.COLOR_BGR2RGB,
            ("RGB", "BGR"): cv2.COLOR_RGB2BGR,
            ("BGR", "GRAY"): cv2.COLOR_BGR2GRAY,
            ("RGB", "GRAY"): cv2.COLOR_RGB2GRAY,
            ("GRAY", "BGR"): cv2.COLOR_GRAY2BGR,
            ("GRAY", "RGB"): cv2.COLOR_GRAY2RGB
        }
        
        conversion_key = (from_space.upper(), to_space.upper())
        if conversion_key in color_conversions:
            return cv2.cvtColor(image, color_conversions[conversion_key])
        else:
            logger.warning(f"Conversão de cor não suportada: {from_space} -> {to_space}")
            return image
    
    @staticmethod
    def create_blank_image(width: int, height: int, color: Tuple[int, int, int] = (0, 0, 0)) -> np.ndarray:
        """Cria uma imagem em branco com as dimensões especificadas."""
        return np.full((height, width, 3), color, dtype=np.uint8)
    
    @staticmethod
    def crop_image(image: np.ndarray, bbox: Tuple[int, int, int, int]) -> np.ndarray:
        """Recorta uma imagem usando bounding box."""
        x1, y1, x2, y2 = bbox
        return image[y1:y2, x1:x2]
