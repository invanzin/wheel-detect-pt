from dataclasses import dataclass
from typing import Tuple, Optional

@dataclass
class Wheel:
    """Classe para armazenar informações de uma roda.
    
    Esta classe foi movida para um arquivo próprio para melhor organização
    e para ser consistente com a estrutura de modelos esperada por outras classes,
    como Axle.
    """
    center: Tuple[int, int]
    bbox: Tuple[int, int, int, int]
    confidence: float
    
    # Campos adicionais para compatibilidade com o modelo Axle mais robusto
    detection_id: Optional[str] = None
    wheel_type: Optional[str] = None
    is_visible: bool = True # Assumir visível por padrão
    
    @property
    def area(self) -> float:
        """Calcula a área da bounding box da roda."""
        x1, y1, x2, y2 = self.bbox
        return float((x2 - x1) * (y2 - y1))
        
    def to_dict(self) -> dict:
        """Converte a roda para dicionário."""
        return {
            'center': self.center,
            'bbox': self.bbox,
            'confidence': self.confidence,
            'detection_id': self.detection_id,
            'wheel_type': self.wheel_type,
            'is_visible': self.is_visible,
            'area': self.area
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'Wheel':
        """Cria uma roda a partir de um dicionário."""
        return cls(
            center=tuple(data['center']),
            bbox=tuple(data['bbox']),
            confidence=data['confidence'],
            detection_id=data.get('detection_id'),
            wheel_type=data.get('wheel_type'),
            is_visible=data.get('is_visible', True)
        )