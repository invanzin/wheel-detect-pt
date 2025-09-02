"""
Modelo para resultado completo da detecção.
"""
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from datetime import datetime
import uuid
from .wheel import Wheel
from .axle import Axle
from ..exceptions.detection_errors import ValidationError
from ...utils.validation import Validator

@dataclass
class DetectionResult:
    """Resultado completo de uma detecção de rodas e eixos."""
    
    # Identificação
    detection_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=datetime.now)
    
    # Dados de entrada
    input_image_path: Optional[Path] = None
    input_image_info: Optional[Dict[str, Any]] = None
    
    # Detecções
    wheels: List[Wheel] = field(default_factory=list)
    axles: List[Axle] = field(default_factory=list)
    
    # Resultados processados
    total_wheels: int = 0
    total_axles: int = 0
    lifted_axles: int = 0
    axles_on_ground: int = 0
    
    # Métricas de qualidade
    average_confidence: float = 0.0
    ground_line_params: Optional[Tuple[float, float]] = None
    
    # Outputs
    output_image_path: Optional[Path] = None
    output_json_path: Optional[Path] = None
    
    # Metadados
    processing_time: Optional[float] = None
    model_version: Optional[str] = None
    processing_notes: Optional[str] = None
    
    def __post_init__(self):
        """Valida e calcula métricas após a inicialização."""
        self._validate_data()
        self._calculate_metrics()
    
    def _validate_data(self):
        """Valida os dados do resultado."""
        try:
            # Validar wheels
            if self.wheels:
                for wheel in self.wheels:
                    if not isinstance(wheel, Wheel):
                        raise ValidationError("Todos os elementos em wheels devem ser do tipo Wheel")
            
            # Validar axles
            if self.axles:
                for axle in self.axles:
                    if not isinstance(axle, Axle):
                        raise ValidationError("Todos os elementos em axles devem ser do tipo Axle")
            
            # Validar confidence
            if self.average_confidence < 0.0 or self.average_confidence > 1.0:
                raise ValidationError("Confiança média deve estar entre 0.0 e 1.0")
                
        except Exception as e:
            if isinstance(e, ValidationError):
                raise
            raise ValidationError(f"Erro na validação do resultado: {e}")
    
    def _calculate_metrics(self):
        """Calcula métricas baseadas nos dados."""
        # Contar rodas e eixos
        self.total_wheels = len(self.wheels)
        self.total_axles = len(self.axles)
        
        # Contar eixos levantados
        self.lifted_axles = sum(1 for axle in self.axles if axle.is_lifted)
        self.axles_on_ground = self.total_axles - self.lifted_axles
        
        # Calcular confiança média
        if self.wheels:
            confidences = [wheel.confidence for wheel in self.wheels]
            self.average_confidence = sum(confidences) / len(confidences)
        elif self.axles:
            confidences = [axle.confidence for axle in self.axles]
            self.average_confidence = sum(confidences) / len(confidences)
    
    def add_wheel(self, wheel: Wheel):
        """Adiciona uma roda ao resultado."""
        if not isinstance(wheel, Wheel):
            raise ValidationError("Objeto deve ser do tipo Wheel")
        
        self.wheels.append(wheel)
        self._calculate_metrics()
    
    def add_axle(self, axle: Axle):
        """Adiciona um eixo ao resultado."""
        if not isinstance(axle, Axle):
            raise ValidationError("Objeto deve ser do tipo Axle")
        
        self.axles.append(axle)
        self._calculate_metrics()
    
    def get_wheel_by_id(self, wheel_id: str) -> Optional[Wheel]:
        """Retorna uma roda pelo ID."""
        for wheel in self.wheels:
            if wheel.detection_id == wheel_id:
                return wheel
        return None
    
    def get_axle_by_id(self, axle_id: str) -> Optional[Axle]:
        """Retorna um eixo pelo ID."""
        for axle in self.axles:
            if axle.axle_id == axle_id:
                return axle
        return None
    
    def get_wheels_by_confidence(self, min_confidence: float) -> List[Wheel]:
        """Retorna rodas com confiança acima do threshold."""
        return [wheel for wheel in self.wheels if wheel.confidence >= min_confidence]
    
    def get_axles_by_type(self, axle_type: str) -> List[Axle]:
        """Retorna eixos de um tipo específico."""
        return [axle for axle in self.axles if axle.axle_type == axle_type]
    
    def get_lifted_axles(self) -> List[Axle]:
        """Retorna apenas eixos levantados."""
        return [axle for axle in self.axles if axle.is_lifted]
    
    def get_ground_axles(self) -> List[Axle]:
        """Retorna apenas eixos no chão."""
        return [axle for axle in self.axles if not axle.is_lifted]
    
    def calculate_detection_quality_score(self) -> float:
        """Calcula um score de qualidade da detecção."""
        if not self.wheels and not self.axles:
            return 0.0
        
        # Fatores de qualidade
        confidence_score = self.average_confidence * 0.4  # 40% do score
        
        # Score baseado no número de detecções (normalizado)
        detection_score = min(self.total_wheels / 10.0, 1.0) * 0.3  # 30% do score
        
        # Score baseado na consistência (eixos vs rodas)
        consistency_score = 0.0
        if self.total_axles > 0 and self.total_wheels > 0:
            expected_wheels_per_axle = self.total_wheels / self.total_axles
            if 0.5 <= expected_wheels_per_axle <= 2.0:  # Range razoável
                consistency_score = 0.3  # 30% do score
        
        return confidence_score + detection_score + consistency_score
    
    def get_summary(self) -> Dict[str, Any]:
        """Retorna um resumo do resultado."""
        return {
            'detection_id': self.detection_id,
            'timestamp': self.timestamp.isoformat(),
            'total_wheels': self.total_wheels,
            'total_axles': self.total_axles,
            'lifted_axles': self.lifted_axles,
            'axles_on_ground': self.axles_on_ground,
            'average_confidence': self.average_confidence,
            'quality_score': self.calculate_detection_quality_score(),
            'processing_time': self.processing_time,
            'model_version': self.model_version
        }
    
    def to_dict(self) -> dict:
        """Converte o resultado para dicionário."""
        return {
            'detection_id': self.detection_id,
            'timestamp': self.timestamp.isoformat(),
            'input_image_path': str(self.input_image_path) if self.input_image_path else None,
            'input_image_info': self.input_image_info,
            'wheels': [wheel.to_dict() for wheel in self.wheels],
            'axles': [axle.to_dict() for axle in self.axles],
            'total_wheels': self.total_wheels,
            'total_axles': self.total_axles,
            'lifted_axles': self.lifted_axles,
            'axles_on_ground': self.axles_on_ground,
            'average_confidence': self.average_confidence,
            'ground_line_params': self.ground_line_params,
            'output_image_path': str(self.output_image_path) if self.output_image_path else None,
            'output_json_path': str(self.output_json_path) if self.output_json_path else None,
            'processing_time': self.processing_time,
            'model_version': self.model_version,
            'processing_notes': self.processing_notes,
            'quality_score': self.calculate_detection_quality_score()
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'DetectionResult':
        """Cria um resultado a partir de um dicionário."""
        # Converter timestamp
        timestamp = datetime.fromisoformat(data['timestamp']) if 'timestamp' in data else datetime.now()
        
        # Converter paths
        input_path = Path(data['input_image_path']) if data.get('input_image_path') else None
        output_path = Path(data['output_image_path']) if data.get('output_image_path') else None
        json_path = Path(data['output_json_path']) if data.get('output_json_path') else None
        
        # Converter wheels e axles
        wheels = [Wheel.from_dict(wheel_data) for wheel_data in data.get('wheels', [])]
        axles = [Axle.from_dict(axle_data) for axle_data in data.get('axles', [])]
        
        return cls(
            detection_id=data.get('detection_id'),
            timestamp=timestamp,
            input_image_path=input_path,
            input_image_info=data.get('input_image_info'),
            wheels=wheels,
            axles=axles,
            ground_line_params=data.get('ground_line_params'),
            output_image_path=output_path,
            output_json_path=json_path,
            processing_time=data.get('processing_time'),
            model_version=data.get('model_version'),
            processing_notes=data.get('processing_notes')
        )
    
    def __str__(self) -> str:
        """Representação string do resultado."""
        return (f"DetectionResult(id={self.detection_id[:8]}, "
                f"wheels={self.total_wheels}, axles={self.total_axles}, "
                f"conf={self.average_confidence:.3f})")
    
    def __repr__(self) -> str:
        """Representação detalhada do resultado."""
        return (f"DetectionResult(detection_id={self.detection_id}, "
                f"timestamp={self.timestamp}, wheels={len(self.wheels)}, "
                f"axles={len(self.axles)})")
