"""
Arquivo de teste para demonstrar os novos modelos de dados.
Execute este arquivo para ver os modelos funcionando.
"""
import sys
from pathlib import Path

# Adicionar o diretório src ao path para imports
sys.path.insert(0, str(Path(__file__).parent))

from core.models import Wheel, Axle, DetectionResult
from core.exceptions.detection_errors import ValidationError
import time

def test_wheel_creation():
    """Testa a criação de uma roda."""
    print("=== Testando criação de roda ===")
    
    try:
        # Criar roda válida
        wheel = Wheel(
            center=(100, 200),
            bbox=(90, 190, 110, 210),
            confidence=0.85,
            wheel_type="dianteira"
        )
        print(f"✅ Roda criada com sucesso: {wheel}")
        print(f"   Área: {wheel.area:.1f}")
        print(f"   Aspect ratio: {wheel.aspect_ratio:.2f}")
        
        # Testar propriedades
        corners = wheel.get_bbox_corners()
        print(f"   Cantos do bbox: {corners}")
        
        # Testar se ponto está dentro
        inside = wheel.is_point_inside((100, 200))
        print(f"   Ponto (100, 200) está dentro: {inside}")
        
        return wheel
        
    except Exception as e:
        print(f"❌ Erro ao criar roda: {e}")
        return None

def test_axle_creation(wheel1, wheel2):
    """Testa a criação de um eixo."""
    print("\n=== Testando criação de eixo ===")
    
    try:
        # Criar eixo com duas rodas
        axle = Axle(
            wheels=[wheel1, wheel2],
            position=(150, 200),
            confidence=0.80,
            axle_type="dianteiro"
        )
        print(f"✅ Eixo criado com sucesso: {axle}")
        print(f"   Número de rodas: {axle.wheel_count}")
        print(f"   É dual: {axle.is_dual}")
        print(f"   Está levantado: {axle.is_lifted}")
        print(f"   Confiança total: {axle.total_confidence:.3f}")
        print(f"   Espaçamento: {axle.calculate_spacing():.1f}")
        
        return axle
        
    except Exception as e:
        print(f"❌ Erro ao criar eixo: {e}")
        return None

def test_detection_result(wheel1, wheel2, axle):
    """Testa a criação de um resultado de detecção."""
    print("\n=== Testando resultado de detecção ===")
    
    try:
        # Criar resultado
        result = DetectionResult(
            wheels=[wheel1, wheel2],
            axles=[axle],
            processing_time=2.5,
            model_version="v1.0"
        )
        
        print(f"✅ Resultado criado com sucesso: {result}")
        print(f"   Total de rodas: {result.total_wheels}")
        print(f"   Total de eixos: {result.total_axles}")
        print(f"   Eixos levantados: {result.lifted_axles}")
        print(f"   Confiança média: {result.average_confidence:.3f}")
        print(f"   Score de qualidade: {result.calculate_detection_quality_score():.3f}")
        
        # Testar métodos
        lifted_axles = result.get_lifted_axles()
        print(f"   Eixos levantados (método): {len(lifted_axles)}")
        
        high_conf_wheels = result.get_wheels_by_confidence(0.8)
        print(f"   Rodas com confiança > 0.8: {len(high_conf_wheels)}")
        
        return result
        
    except Exception as e:
        print(f"❌ Erro ao criar resultado: {e}")
        return None

def test_serialization(result):
    """Testa serialização e deserialização."""
    print("\n=== Testando serialização ===")
    
    try:
        # Converter para dicionário
        result_dict = result.to_dict()
        print(f"✅ Conversão para dicionário: {len(result_dict)} campos")
        
        # Converter de volta para objeto
        result_recovered = DetectionResult.from_dict(result_dict)
        print(f"✅ Recuperação do dicionário: {result_recovered}")
        
        # Verificar se são iguais
        if (result.detection_id == result_recovered.detection_id and
            result.total_wheels == result_recovered.total_wheels):
            print("✅ Serialização/deserialização funcionou corretamente!")
        else:
            print("❌ Problema na serialização/deserialização")
            
    except Exception as e:
        print(f"❌ Erro na serialização: {e}")

def test_validation_errors():
    """Testa validações de erro."""
    print("\n=== Testando validações de erro ===")
    
    try:
        # Tentar criar roda com bbox inválido
        Wheel(
            center=(100, 200),
            bbox=(110, 210, 90, 190),  # x1 > x2, y1 > y2
            confidence=0.85
        )
        print("❌ Deveria ter falhado com bbox inválido")
        
    except ValidationError as e:
        print(f"✅ Validação funcionou: {e}")
    
    try:
        # Tentar criar roda com confiança inválida
        Wheel(
            center=(100, 200),
            bbox=(90, 190, 110, 210),
            confidence=1.5  # > 1.0
        )
        print("❌ Deveria ter falhado com confiança inválida")
        
    except ValidationError as e:
        print(f"✅ Validação funcionou: {e}")

def main():
    """Função principal de teste."""
    print("🚗 Testando modelos de dados do sistema de detecção de rodas")
    print("=" * 60)
    
    # Testar criação de rodas
    wheel1 = test_wheel_creation()
    if not wheel1:
        return
    
    # Criar segunda roda
    wheel2 = Wheel(
        center=(200, 200),
        bbox=(190, 190, 210, 210),
        confidence=0.90,
        wheel_type="dianteira"
    )
    
    # Testar criação de eixo
    axle = test_axle_creation(wheel1, wheel2)
    if not axle:
        return
    
    # Testar resultado de detecção
    result = test_detection_result(wheel1, wheel2, axle)
    if not result:
        return
    
    # Testar serialização
    test_serialization(result)
    
    # Testar validações de erro
    test_validation_errors()
    
    print("\n" + "=" * 60)
    print("🎉 Todos os testes concluídos com sucesso!")
    print("Os modelos estão funcionando corretamente.")

if __name__ == "__main__":
    main()
