"""
Utilitários matemáticos para cálculos geométricos e estatísticos.
"""
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from sklearn.linear_model import RANSACRegressor, LinearRegression
import logging

logger = logging.getLogger(__name__)

class MathUtils:
    """Utilitários matemáticos para cálculos geométricos."""
    
    @staticmethod
    def calculate_distance(point1: Tuple[float, float], point2: Tuple[float, float]) -> float:
        """Calcula a distância euclidiana entre dois pontos."""
        return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)
    
    @staticmethod
    def calculate_center(points: List[Tuple[float, float]]) -> Tuple[float, float]:
        """Calcula o centro (centroide) de uma lista de pontos."""
        if not points:
            return (0.0, 0.0)
        
        x_coords = [p[0] for p in points]
        y_coords = [p[1] for p in points]
        
        center_x = np.mean(x_coords)
        center_y = np.mean(y_coords)
        
        return (center_x, center_y)
    
    @staticmethod
    def fit_line_ransac(points: List[Tuple[float, float]], 
                        min_samples: int = 2,
                        residual_threshold: float = 1.0) -> Optional[Tuple[float, float]]:
        """
        Ajusta uma linha usando RANSAC para regressão robusta.
        
        Returns:
            Tuple (slope, intercept) ou None se falhar
        """
        try:
            if len(points) < min_samples:
                logger.warning(f"Poucos pontos para ajustar linha: {len(points)} < {min_samples}")
                return None
            
            # Converter pontos para arrays numpy
            points_array = np.array(points)
            x = points_array[:, 0].reshape(-1, 1)
            y = points_array[:, 1]
            
            # Ajustar linha usando RANSAC
            ransac = RANSACRegressor(
                min_samples=min_samples,
                residual_threshold=residual_threshold,
                random_state=42
            )
            
            ransac.fit(x, y)
            
            # Extrair parâmetros da linha
            slope = ransac.estimator_.coef_[0]
            intercept = ransac.estimator_.intercept_
            
            logger.info(f"Linha ajustada com RANSAC: slope={slope:.3f}, intercept={intercept:.3f}")
            return (slope, intercept)
            
        except Exception as e:
            logger.error(f"Erro ao ajustar linha com RANSAC: {e}")
            return None
    
    @staticmethod
    def fit_line_linear(points: List[Tuple[float, float]]) -> Optional[Tuple[float, float]]:
        """
        Ajusta uma linha usando regressão linear simples.
        
        Returns:
            Tuple (slope, intercept) ou None se falhar
        """
        try:
            if len(points) < 2:
                logger.warning("Poucos pontos para ajustar linha linear")
                return None
            
            # Converter pontos para arrays numpy
            points_array = np.array(points)
            x = points_array[:, 0].reshape(-1, 1)
            y = points_array[:, 1]
            
            # Ajustar linha usando regressão linear
            lr = LinearRegression()
            lr.fit(x, y)
            
            # Extrair parâmetros da linha
            slope = lr.coef_[0]
            intercept = lr.intercept_
            
            logger.info(f"Linha ajustada com regressão linear: slope={slope:.3f}, intercept={intercept:.3f}")
            return (slope, intercept)
            
        except Exception as e:
            logger.error(f"Erro ao ajustar linha com regressão linear: {e}")
            return None
    
    @staticmethod
    def calculate_point_to_line_distance(point: Tuple[float, float], 
                                      slope: float, 
                                      intercept: float) -> float:
        """Calcula a distância de um ponto a uma linha."""
        x, y = point
        
        # Fórmula da distância ponto-linha: |ax + by + c| / sqrt(a² + b²)
        # Para linha y = mx + b, temos: -mx + y - b = 0
        # Então: a = -m, b = 1, c = -b
        a = -slope
        b = 1
        c = -intercept
        
        distance = abs(a * x + b * y + c) / np.sqrt(a**2 + b**2)
        return distance
    
    @staticmethod
    def group_points_by_proximity(points: List[Tuple[float, float]], 
                                 threshold: float) -> List[List[Tuple[float, float]]]:
        """
        Agrupa pontos por proximidade usando um threshold.
        
        Args:
            points: Lista de pontos (x, y)
            threshold: Distância máxima para considerar pontos próximos
            
        Returns:
            Lista de grupos de pontos
        """
        if not points:
            return []
        
        groups = []
        used = set()
        
        for i, point in enumerate(points):
            if i in used:
                continue
            
            # Criar novo grupo
            group = [point]
            used.add(i)
            
            # Procurar pontos próximos
            for j, other_point in enumerate(points):
                if j in used:
                    continue
                
                if MathUtils.calculate_distance(point, other_point) <= threshold:
                    group.append(other_point)
                    used.add(j)
            
            groups.append(group)
        
        return groups
    
    @staticmethod
    def calculate_confidence_score(predictions: List[float], 
                                 weights: Optional[List[float]] = None) -> float:
        """
        Calcula um score de confiança ponderado.
        
        Args:
            predictions: Lista de valores de confiança
            weights: Pesos opcionais para cada predição
            
        Returns:
            Score de confiança ponderado
        """
        if not predictions:
            return 0.0
        
        if weights is None:
            weights = [1.0] * len(predictions)
        
        if len(predictions) != len(weights):
            logger.warning("Número de predições e pesos não coincidem")
            weights = [1.0] * len(predictions)
        
        # Calcular média ponderada
        weighted_sum = sum(p * w for p, w in zip(predictions, weights))
        total_weight = sum(weights)
        
        if total_weight == 0:
            return 0.0
        
        return weighted_sum / total_weight
    
    @staticmethod
    def normalize_coordinates(points: List[Tuple[float, float]], 
                            target_range: Tuple[float, float] = (0.0, 1.0)) -> List[Tuple[float, float]]:
        """
        Normaliza coordenadas para um intervalo específico.
        
        Args:
            points: Lista de pontos (x, y)
            target_range: Intervalo de destino (min, max)
            
        Returns:
            Lista de pontos normalizados
        """
        if not points:
            return []
        
        # Encontrar limites atuais
        x_coords = [p[0] for p in points]
        y_coords = [p[1] for p in points]
        
        x_min, x_max = min(x_coords), max(x_coords)
        y_min, y_max = min(y_coords), max(y_coords)
        
        # Evitar divisão por zero
        x_range = x_max - x_min if x_max != x_min else 1.0
        y_range = y_max - y_min if y_max != y_min else 1.0
        
        target_min, target_max = target_range
        target_range_size = target_max - target_min
        
        # Normalizar pontos
        normalized_points = []
        for x, y in points:
            norm_x = target_min + (x - x_min) / x_range * target_range_size
            norm_y = target_min + (y - y_min) / y_range * target_range_size
            normalized_points.append((norm_x, norm_y))
        
        return normalized_points
