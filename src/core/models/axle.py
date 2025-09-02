"""
Modelo de dados para um eixo detectado.
"""
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any
import numpy as np
from .wheel import Wheel
from ..exceptions.detection_errors import ValidationError
from ...utils.validation import Validator
from ...utils.math_utils import MathUtils

@dataclass
class Axle:
    """Classe para armazenar informações de um eixo detectado."""
    
    wheels: List[Wheel]
    position: Tuple[float, float]
    confidence: float
    
    # Propriedades do eixo
    is_dual: bool = False
    is_lifted: bool = False
    axle_type: Optional[str] = None
    
    # Metadados
    axle_id: Optional[str] = None
    detection_timestamp: Optional[float] = None
    
    # Configurações
    grouping_threshold: float = 40.0
    lifted_threshold: float = 5.0
    
    def __post_init__(self):
        """Valida os dados após a inicialização."""
        self._validate_data()
        self._analyze_axle_properties()
    
    def _validate_data(self):
        """Valida os dados do eixo."""
        try:
            # Validar wheels
            Validator.validate_list_not_empty(self.wheels, "rodas do eixo")
            
            # Validar position
            Validator.validate_coordinates([self.position], min_points=1, name="posição do eixo")
            
            # Validar confidence
            self.confidence = Validator.validate_confidence_threshold(self.confidence, "confiança do eixo")
            
            # Validar thresholds
            self.grouping_threshold = Validator.validate_numeric_range(
                self.grouping_threshold, min_value=0.0, name="threshold de agrupamento"
            )
            self.lifted_threshold = Validator.validate_numeric_range(
                self.lifted_threshold, min_value=0.0, name="threshold de eixo levantado"
            )
            
        except Exception as e:
            if isinstance(e, ValidationError):
                raise
            raise ValidationError(f"Erro na validação do eixo: {e}")
    
    def _analyze_axle_properties(self):
        """Analisa e define as propriedades do eixo."""
        if len(self.wheels) > 1:
            self.is_dual = True
        
        # Determinar se o eixo está levantado baseado na posição das rodas
        self._determine_lifted_status()
    
    def _determine_lifted_status(self):
        """Determina se o eixo está levantado."""
        if not self.wheels:
            return
        
        # Calcular a posição média das rodas
        wheel_positions = [wheel.center for wheel in self.wheels]
        avg_position = MathUtils.calculate_center(wheel_positions)
        
        # Comparar com a posição do eixo
        distance = MathUtils.calculate_distance(avg_position, self.position)
        
        # Se a distância for maior que o threshold, considerar levantado
        self.is_lifted = distance > self.lifted_threshold
    
    @property
    def wheel_count(self) -> int:
        """Retorna o número de rodas no eixo."""
        return len(self.wheels)
    
    @property
    def total_confidence(self) -> float:
        """Retorna a confiança total do eixo (média das rodas)."""
        if not self.wheels:
            return 0.0
        
        confidences = [wheel.confidence for wheel in self.wheels]
        return MathUtils.calculate_confidence_score(confidences)
    
    @property
    def bounding_box(self) -> Tuple[float, float, float, float]:
        """Retorna o bounding box que engloba todas as rodas do eixo."""
        if not self.wheels:
            return (0.0, 0.0, 0.0, 0.0)
        
        # Encontrar limites
        x_coords = []
        y_coords = []
        
        for wheel in self.wheels:
            x1, y1, x2, y2 = wheel.bbox
            x_coords.extend([x1, x2])
            y_coords.extend([y1, y2])
        
        x_min, x_max = min(x_coords), max(x_coords)
        y_min, y_max = min(y_coords), max(y_coords)
        
        return (x_min, y_min, x_max, y_max)
    
    @property
    def area(self) -> float:
        """Calcula a área total do eixo."""
        if not self.wheels:
            return 0.0
        
        total_area = sum(wheel.area for wheel in self.wheels)
        return total_area
    
    @property
    def center_of_mass(self) -> Tuple[float, float]:
        """Calcula o centro de massa do eixo baseado nas rodas."""
        if not self.wheels:
            return self.position
        
        wheel_positions = [wheel.center for wheel in self.wheels]
        return MathUtils.calculate_center(wheel_positions)
    
    def add_wheel(self, wheel: Wheel):
        """Adiciona uma roda ao eixo."""
        if not isinstance(wheel, Wheel):
            raise ValidationError("Objeto deve ser do tipo Wheel")
        
        self.wheels.append(wheel)
        self._analyze_axle_properties()
    
    def remove_wheel(self, wheel: Wheel):
        """Remove uma roda do eixo."""
        if wheel in self.wheels:
            self.wheels.remove(wheel)
            self._analyze_axle_properties()
    
    def get_wheel_by_id(self, wheel_id: str) -> Optional[Wheel]:
        """Retorna uma roda pelo ID."""
        for wheel in self.wheels:
            if wheel.detection_id == wheel_id:
                return wheel
        return None
    
    def get_wheels_by_type(self, wheel_type: str) -> List[Wheel]:
        """Retorna rodas de um tipo específico."""
        return [wheel for wheel in self.wheels if wheel.wheel_type == wheel_type]
    
    def get_visible_wheels(self) -> List[Wheel]:
        """Retorna apenas rodas visíveis."""
        return [wheel for wheel in self.wheels if wheel.is_visible]
    
    def calculate_spacing(self) -> float:
        """Calcula o espaçamento entre rodas do eixo (para eixos duais)."""
        if len(self.wheels) < 2:
            return 0.0
        
        # Calcular distância média entre rodas
        distances = []
        for i in range(len(self.wheels)):
            for j in range(i + 1, len(self.wheels)):
                dist = MathUtils.calculate_distance(
                    self.wheels[i].center, 
                    self.wheels[j].center
                )
                distances.append(dist)
        
        if not distances:
            return 0.0
        
        return np.mean(distances)
    
    def is_near_point(self, point: Tuple[float, float], threshold: float) -> bool:
        """Verifica se o eixo está próximo a um ponto."""
        distance = MathUtils.calculate_distance(self.position, point)
        return distance <= threshold
    
    def get_overlap_with_axle(self, other_axle: 'Axle') -> float:
        """Calcula a sobreposição com outro eixo."""
        if not self.wheels or not other_axle.wheels:
            return 0.0
        
        # Calcular sobreposição média entre rodas
        overlaps = []
        for wheel1 in self.wheels:
            for wheel2 in other_axle.wheels:
                overlap = wheel1.get_overlap_with_other(wheel2)
                overlaps.append(overlap)
        
        if not overlaps:
            return 0.0
        
        return np.mean(overlaps)
    
    def to_dict(self) -> dict:
        """Converte o eixo para dicionário."""
        return {
            'wheels': [wheel.to_dict() for wheel in self.wheels],
            'position': self.position,
            'confidence': self.confidence,
            'is_dual': self.is_dual,
            'is_lifted': self.is_lifted,
            'axle_type': self.axle_type,
            'axle_id': self.axle_id,
            'detection_timestamp': self.detection_timestamp,
            'wheel_count': self.wheel_count,
            'total_confidence': self.total_confidence,
            'area': self.area,
            'center_of_mass': self.center_of_mass,
            'spacing': self.calculate_spacing()
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'Axle':
        """Cria um eixo a partir de um dicionário."""
        required_fields = ['wheels', 'position', 'confidence']
        for field in required_fields:
            if field not in data:
                raise ValidationError(f"Campo obrigatório ausente: {field}")
        
        # Converter wheels de volta para objetos Wheel
        wheels = [Wheel.from_dict(wheel_data) for wheel_data in data['wheels']]
        
        return cls(
            wheels=wheels,
            position=data['position'],
            confidence=data['confidence'],
            is_dual=data.get('is_dual', False),
            is_lifted=data.get('is_lifted', False),
            axle_type=data.get('axle_type'),
            axle_id=data.get('axle_id'),
            detection_timestamp=data.get('detection_timestamp')
        )
    
    def __str__(self) -> str:
        """Representação string do eixo."""
        status = "Levantado" if self.is_lifted else "No chão"
        return f"Axle({self.wheel_count} rodas, {status}, conf={self.confidence:.3f})"
    
    def __repr__(self) -> str:
        """Representação detalhada do eixo."""
        return (f"Axle(wheels={len(self.wheels)}, position={self.position}, "
                f"confidence={self.confidence:.3f}, lifted={self.is_lifted})")
