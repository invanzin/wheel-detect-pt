from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import logging
import cv2
import numpy as np
from ultralytics import YOLO
import argparse
import json
import shutil
from datetime import datetime
from sklearn.linear_model import RANSACRegressor, LinearRegression  # regressão robusta

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class Wheel:
    """Classe para armazenar informações de uma roda."""
    center: Tuple[int, int]
    bbox: Tuple[int, int, int, int]
    confidence: float

@dataclass
class Axle:
    """Classe para armazenar informações de um eixo."""
    wheels: List[Wheel]
    is_dual: bool
    is_lifted: bool
    position: Tuple[int, int]  # posição média do eixo (x, y)
    confidence: float

class WheelDetector:
    """Classe para detecção de rodas e análise de eixos."""
    
    def __init__(
        self,
        model_path: str = 'wheel-detector-v11n-finetuned/weights/best.pt',
        conf_threshold: float = 0.55,
        axle_grouping_threshold: int = 40,
        lifted_axle_threshold: int = 5,
    ):
        """
        Inicializa o detector de rodas.
        
        Args:
            model_path: Caminho para o modelo YOLO.
            conf_threshold: Confiança mínima para detecções (0.75 = 75%).
            axle_grouping_threshold: Tolerância vertical para agrupar rodas no mesmo eixo.
            lifted_axle_threshold: Tolerância vertical para identificar eixos levantados.
        """
        self.model = self._load_model(model_path)
        self.conf_threshold = conf_threshold
        self.axle_grouping_threshold = axle_grouping_threshold
        self.lifted_axle_threshold = lifted_axle_threshold
    
    def _load_model(self, model_path: str) -> YOLO:
        """Carrega o modelo YOLO."""
        try:
            model = YOLO(model_path)
            logger.info(f"Modelo carregado com sucesso: {model_path}")
            return model
        except Exception as e:
            logger.error(f"Erro ao carregar o modelo: {e}")
            raise
    
    def _detect_wheels(self, image: np.ndarray) -> List[Wheel]:
        """
        Detecta rodas na imagem usando YOLO.
        
        Args:
            image: Imagem em formato numpy array
            
        Returns:
            Lista de objetos Wheel detectados
        """
        results = self.model.predict(
            source=image,
            imgsz=736,
            conf=self.conf_threshold,
            show=False
        )
        
        wheels = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = [int(coord) for coord in box.xyxy[0].tolist()]
                confidence = float(box.conf[0])
                
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2
                
                wheels.append(Wheel(
                    center=(center_x, center_y),
                    bbox=(x1, y1, x2, y2),
                    confidence=confidence
                ))
        
        # Ordenar rodas da esquerda para a direita para facilitar o agrupamento
        wheels.sort(key=lambda w: w.center[0])
        return wheels
    
    def _infer_ground_line(self,     wheels: List[Wheel]) -> Tuple[float, float]:
        """
        Infere a linha do chão usando TODAS as rodas detectadas.
        
        Passos:
        1. Usa TODAS as rodas para calcular a linha do chão, sem fazer
           suposições sobre quais estão no chão ou levantadas.
        2. Ajusta uma regressão linear robusta (RANSAC) sobre todos os
           pontos para obter a reta do chão. Se todas as rodas tiverem
           mesma coordenada *x* (caso extremo) a reta é tratada como
           horizontal.

        Returns:
            Tuple[slope, intercept]: parâmetros da linha y = slope*x + intercept
        """
        if len(wheels) < 2:
            # Se temos apenas uma roda, assumir linha horizontal passando pela
            # sua borda inferior.
            return 0.0, wheels[0].bbox[3] if wheels else 0.0

        # -------------------------------------------------------------
        # 1) Usar TODAS as rodas para calcular a linha do chão
        # -------------------------------------------------------------
        # Não fazemos mais suposições sobre quais rodas estão no chão
        # Usamos todas as rodas para uma regressão mais robusta

        # -------------------------------------------------------------
        # 2) Ajustar reta y = slope * x + intercept usando TODAS as rodas
        # -------------------------------------------------------------
        xs = np.array([w.center[0] for w in wheels], dtype=np.float32)
        ys = np.array([w.bbox[3] for w in wheels], dtype=np.float32)

        if len(np.unique(xs)) == 1:
            # Pontos todos com o mesmo X → reta horizontal
            slope = 0.0
            intercept = float(np.median(ys))
        else:
            # -----------------------------------------------------
            # Ajuste robusto via RANSAC (ignora outliers)
            # -----------------------------------------------------
            try:
                xs_2d = xs.reshape(-1, 1)
                model_ransac = RANSACRegressor(
                    base_estimator=LinearRegression(),
                    residual_threshold=5.0,  # px – ajuste se necessário
                    max_trials=100,
                    random_state=42,
                )
                model_ransac.fit(xs_2d, ys)
                slope = float(model_ransac.estimator_.coef_[0])
                intercept = float(model_ransac.estimator_.intercept_)
                logger.info(
                    f"RANSAC inliers: {model_ransac.inlier_mask_.sum()} / {len(xs)}"
                )
            except Exception as e:
                # Fallback para polyfit caso RANSAC falhe
                logger.warning(f"RANSAC falhou ({e}), usando polyfit.")
                slope, intercept = np.polyfit(xs, ys, 1)

        logger.info(
            f"Linha do chão (RANSAC): y = {slope:.4f}x + {intercept:.2f} "
            f"[baseado em {len(wheels)} rodas]"
        )
        return slope, intercept

    def _calculate_distance_to_ground_line(self, wheel: Wheel, slope: float, intercept: float) -> float:
        """
        Calcula a distância perpendicular de uma roda até a linha do chão.
        
        Args:
            wheel: Roda para calcular a distância
            slope: Inclinação da linha do chão
            intercept: Intercepto da linha do chão
            
        Returns:
            Distância perpendicular (positiva se a roda está acima da linha)
        """
        # Ponto da roda (centro horizontal, parte inferior vertical)
        x_wheel = wheel.center[0]
        y_wheel = wheel.bbox[3]
        
        # Linha do chão: y = slope*x + intercept
        # Forma geral: slope*x - y + intercept = 0
        # Distância perpendicular = |ax + by + c| / sqrt(a² + b²)
        a = slope
        b = -1
        c = intercept
        
        distance = abs(a * x_wheel + b * y_wheel + c) / np.sqrt(a**2 + b**2)
        
        # Determinar se a roda está acima ou abaixo da linha
        y_ground_at_wheel = slope * x_wheel + intercept
        if y_wheel < y_ground_at_wheel:
            # Roda está acima da linha do chão (levantada)
            return distance
        else:
            # Roda está abaixo ou na linha do chão
            return -distance

    def _group_wheels_into_axles(self, wheels: List[Wheel]) -> Tuple[List[Axle], Tuple[float, float]]:
        """
        Trata cada roda detectada como um eixo individual e verifica se está levantada
        usando uma linha diagonal inferida do chão.
        """
        if not wheels:
            return [], (0.0, 0.0)

        # Inferir a linha do chão considerando a perspectiva diagonal
        slope, intercept = self._infer_ground_line(wheels)

        # -------------------------------------------------------------
        # Calcular threshold dinâmico para detectar rodas levantadas
        # -------------------------------------------------------------
        # Diâmetro de cada roda (altura da bounding box)
        diameters = np.array([w.bbox[3] - w.bbox[1] for w in wheels], dtype=np.float32)
        avg_diameter = float(np.mean(diameters)) if len(diameters) else 0.0

        # 15 % do diâmetro médio ou o valor fixo fornecido — o que for maior
        dynamic_threshold = max(self.lifted_axle_threshold, avg_diameter * 0.05)

        logger.info(
            f"Threshold dinâmico para eixo levantado: {dynamic_threshold:.2f}px "
            f"(diametro médio {avg_diameter:.2f}px)"
        )

        # Criar um eixo para cada roda
        final_axles = []
        for i, wheel in enumerate(wheels):
            # Calcular distância perpendicular até a linha do chão
            distance_to_ground = self._calculate_distance_to_ground_line(wheel, slope, intercept)
            
            # Roda está levantada se a distância for maior que o threshold
            is_lifted = distance_to_ground > dynamic_threshold
            
            # Log para debug
            logger.info(f"Roda {i+1}: posição {wheel.center}, distância ao chão: {distance_to_ground:.2f}px, levantada: {is_lifted}")
            
            final_axles.append(Axle(
                wheels=[wheel],
                is_dual=False,  # Não considerar eixos duplos
                is_lifted=is_lifted,
                position=wheel.center,
                confidence=wheel.confidence
            ))
        
        final_axles.sort(key=lambda a: a.position[0])
        logger.info(f"Criados {len(final_axles)} eixos. Linha do chão: y = {slope:.4f}x + {intercept:.2f}")
        return final_axles, (slope, intercept)
    
    def _draw_results(self, image: np.ndarray, axles: List[Axle], ground_line_params: Tuple[float, float]) -> np.ndarray:
        """
        Desenha os resultados na imagem.
        
        Args:
            image: Imagem original
            axles: Lista de eixos detectados
            ground_line_params: Parâmetros da linha do chão (slope, intercept)
            
        Returns:
            Imagem com as detecções desenhadas
        """
        image_with_detections = image.copy()
        slope, intercept = ground_line_params
        
        # Desenhar linha do chão inferida (diagonal)
        # Usar apenas a região entre as rodas para melhor visualização
        height, width = image_with_detections.shape[:2]
        
        # Encontrar limites das rodas para desenhar linha apenas na região relevante
        if axles:
            sorted_axles = sorted(axles, key=lambda a: a.position[0])
            x1 = max(0, sorted_axles[0].position[0] - 100)  # Estender um pouco antes da primeira roda
            x2 = min(width, sorted_axles[-1].position[0] + 100)  # Estender um pouco depois da última roda
        else:
            x1, x2 = 0, width
            
        y1 = int(slope * x1 + intercept)
        y2 = int(slope * x2 + intercept)
        
        # Debug: mostrar pontos de referência
        logger.info(f"Desenhando linha: ({x1}, {y1}) -> ({x2}, {y2})")
        logger.info(f"Dimensões da imagem: {width}x{height}")
        
        # Garantir que os pontos estejam dentro da imagem
        y1 = max(0, min(height-1, y1))
        y2 = max(0, min(height-1, y2))
        
        cv2.line(image_with_detections, (x1, y1), (x2, y2), (255, 255, 0), 3)
        cv2.putText(image_with_detections, "Linha do Chao Inferida", (10, y1 - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        # Desenhar cada eixo
        for i, axle in enumerate(axles):
            # Cor baseada no estado do eixo
            color = (0, 0, 255) if axle.is_lifted else (0, 255, 0)

            # Rótulo simples do eixo
            label = f"Eixo {i+1}"
            cv2.putText(
                image_with_detections,
                label,
                (axle.position[0] - 30, axle.position[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2
            )

            # Desenhar rodas
            for wheel in axle.wheels:
                # Bounding box
                cv2.rectangle(
                    image_with_detections,
                    (wheel.bbox[0], wheel.bbox[1]),
                    (wheel.bbox[2], wheel.bbox[3]),
                    (255, 0, 0),
                    2
                )

                # Centro da roda
                cv2.circle(
                    image_with_detections,
                    wheel.center,
                    3,
                    (0, 0, 255),
                    -1
                )
        
        return image_with_detections
    
    def process_image(self, image_path: str, output_dir: str = "output", show: bool = True) -> Dict:
        """
        Processa uma imagem para detectar rodas e eixos.
        
        Args:
            image_path: Caminho para a imagem.
            output_dir: Diretório para salvar os resultados (imagem e JSON).
            show: Se True, exibe a imagem com as detecções em uma janela.
            
        Returns:
            Dicionário com resultados e estatísticas.
        """
        # Verificar se o arquivo existe
        if not Path(image_path).exists():
            raise FileNotFoundError(f"Imagem não encontrada: {image_path}")
        
        # Carregar imagem
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Erro ao carregar a imagem: {image_path}")
        
        # Detectar rodas
        wheels = self._detect_wheels(image)
        logger.info(f"Detectadas {len(wheels)} rodas")
        
        # Agrupar em eixos
        axles, ground_line_params = self._group_wheels_into_axles(wheels)
        logger.info(f"Detectados {len(axles)} eixos")
        
        # Desenhar resultados
        image_with_detections = self._draw_results(image, axles, ground_line_params)
        
        # Criar diretório para salvar resultados
        output_path_dir = Path(output_dir)
        output_path_dir.mkdir(parents=True, exist_ok=True)
        
        # Gerar nome do arquivo baseado no timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        image_name = Path(image_path).name
        output_path = output_path_dir / f"{Path(image_name).stem}_{timestamp}{Path(image_path).suffix}"
        
        # Salvar imagem com detecções
        cv2.imwrite(str(output_path), image_with_detections)
        logger.info(f"Imagem com detecções salva em: {output_path}")
        
        # Mostrar resultado, se solicitado
        if show:
            cv2.imshow('Detecção de Rodas e Eixos', image_with_detections)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        
        # Preparar resultados (convertendo tipos numpy para tipos Python nativos)
        results = {
            'total_axles': int(len(axles)),
            'lifted_axles': int(sum(1 for axle in axles if axle.is_lifted)),
            'total_wheels': int(sum(len(axle.wheels) for axle in axles)),
            'output_path': str(output_path),
            'ground_line_slope': float(ground_line_params[0]),
            'ground_line_intercept': float(ground_line_params[1]),
            'axles': [
                {
                    'index': int(i),
                    'wheels': int(len(axle.wheels)),
                    'is_lifted': bool(axle.is_lifted),
                    'confidence': float(axle.confidence),
                    'position': [int(axle.position[0]), int(axle.position[1])]
                }
                for i, axle in enumerate(axles)
            ],
            # Novo campo com detalhes mais ricos de cada eixo
            'axle_details': [
                {
                    'axle_number': int(i + 1),  # 1-based para melhor legibilidade
                    'wheels': int(len(axle.wheels)),
                    'confidence': float(axle.confidence),
                    'status': 'Levantado' if axle.is_lifted else 'No Chao',
                    'position': [int(axle.position[0]), int(axle.position[1])]
                }
                for i, axle in enumerate(axles)
            ],
            'axles_on_ground': int(len(axles) - sum(1 for axle in axles if axle.is_lifted)),
            'average_confidence': float(np.mean([axle.confidence for axle in axles])) if axles else 0.0
        }
        
        # Salvar resultados em JSON
        json_results_path = output_path_dir / f"{Path(image_name).stem}_{timestamp}_results.json"
        with open(json_results_path, 'w') as f:
            json.dump(results, f, indent=4)
        logger.info(f"Resultados salvos em: {json_results_path}")
        
        # Adicionar o caminho do JSON aos resultados
        results['json_results_path'] = str(json_results_path)
        
        return results

def get_available_categories():
    """Retorna as categorias disponíveis baseadas nas pastas existentes."""
    output_dir = Path("output")
    categories = []
    
    if output_dir.exists():
        for item in output_dir.iterdir():
            if item.is_dir():
                categories.append(item.name)
    
    # Adicionar categorias padrão se não existirem
    default_categories = [
        "Detecção_Rodas",
        "Falha_Detecção_Rodas", 
        "Eixo_Levantado",
        "Falha_Detecção_Eixos_Levantados",
        "Lateral_Cam"
    ]
    
    for cat in default_categories:
        if cat not in categories:
            categories.append(cat)
    
    return sorted(categories)

def select_output_category():
    """Permite ao usuário selecionar a categoria de output."""
    categories = get_available_categories()
    
    print("\n" + "="*50)
    print("CATEGORIAS DISPONÍVEIS:")
    print("="*50)
    
    for i, category in enumerate(categories, 1):
        print(f"{i:2d}. {category}")
    
    print(f"{len(categories) + 1:2d}. Criar nova categoria")
    print(f"{len(categories) + 2:2d}. Salvar na pasta 'output' (padrão)")
    
    while True:
        try:
            choice = input(f"\nEscolha uma categoria (1-{len(categories) + 2}): ").strip()
            choice_num = int(choice)
            
            if 1 <= choice_num <= len(categories):
                selected_category = categories[choice_num - 1]
                
                # Lógica especial para Lateral_Cam
                if selected_category == "Lateral_Cam":
                    return select_lateral_cam_subcategory()
                
                return selected_category
            elif choice_num == len(categories) + 1:
                new_category = input("Digite o nome da nova categoria: ").strip()
                if new_category:
                    return new_category
                else:
                    print("Nome da categoria não pode estar vazio!")
            elif choice_num == len(categories) + 2:
                return "output"
            else:
                print(f"Por favor, escolha um número entre 1 e {len(categories) + 2}")
        except ValueError:
            print("Por favor, digite um número válido!")
        except KeyboardInterrupt:
            print("\nOperação cancelada pelo usuário.")
            return "output"

def select_lateral_cam_subcategory():
    """Permite ao usuário selecionar a subcategoria da Lateral_Cam."""
    print("\n" + "="*30)
    print("SUBCATEGORIA LATERAL_CAM:")
    print("="*30)
    print("1. Lateral_Cam/1")
    print("2. Lateral_Cam/2")
    print("3. Voltar")
    
    while True:
        try:
            choice = input("\nEscolha uma subcategoria (1-3): ").strip()
            choice_num = int(choice)
            
            if choice_num == 1:
                return "Lateral_Cam/1"
            elif choice_num == 2:
                return "Lateral_Cam/2"
            elif choice_num == 3:
                return select_output_category()  # Volta ao menu principal
            else:
                print("Por favor, escolha um número entre 1 e 3")
        except ValueError:
            print("Por favor, digite um número válido!")
        except KeyboardInterrupt:
            print("\nOperação cancelada pelo usuário.")
            return "output"

def move_results_to_category(results_path, json_results_path, category):
    """Move os arquivos de resultado para a categoria selecionada."""
    output_dir = Path("output")
    
    # Lidar com subpastas (ex: Lateral_Cam/1)
    if "/" in category:
        main_category, subcategory = category.split("/", 1)
        category_dir = output_dir / main_category / subcategory
    else:
        category_dir = output_dir / category
    
    # Criar diretório da categoria se não existir (incluindo subpastas)
    category_dir.mkdir(parents=True, exist_ok=True)
    
    # Mover arquivos
    files_moved = []
    
    # Mover imagem anotada
    if Path(results_path).exists():
        new_image_path = category_dir / Path(results_path).name
        shutil.move(str(results_path), str(new_image_path))
        files_moved.append(str(new_image_path))
    
    # Mover arquivo JSON
    if Path(json_results_path).exists():
        new_json_path = category_dir / Path(json_results_path).name
        shutil.move(str(json_results_path), str(new_json_path))
        files_moved.append(str(new_json_path))
    
    return files_moved

def main():
    """Função principal para executar o detector de rodas a partir da linha de comando."""
    parser = argparse.ArgumentParser(description="Detecta rodas em imagens, agrupa em eixos e identifica eixos suspensos.")
    
    parser.add_argument(
        "--image-path",
        type=str,
        required=False,
        default="images\caminhao\sc_caminhao26_up.jpg",
        help="Caminho para a imagem de entrada a ser processada. Se omitido, usa o valor padrão definido no código."
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,  # Usará o padrão da classe WheelDetector se não for fornecido
        help="Caminho para o modelo YOLO. O padrão é o modelo fine-tuned."
    )
    parser.add_argument(
        "--conf-threshold",
        type=float,
        default=0.5, # Usará o padrão da classe WheelDetector se não for fornecido
        help="Confiança mínima para detecção de rodas (ex: 0.3)."
    )
    parser.add_argument(
        "--save-path",
        type=str,
        default="output",
        help="Diretório onde os resultados (imagem e JSON) serão salvos."
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Se incluído, não exibirá a janela com a imagem do resultado."
    )
    parser.add_argument(
        "--auto-categorize",
        action="store_true",
        help="Se incluído, não perguntará a categoria e salvará diretamente na pasta output."
    )
    
    args = parser.parse_args()

    try:
        # Preparar argumentos para o detector.
        # Apenas passamos os argumentos se eles foram fornecidos na linha de comando.
        # Caso contrário, a classe usará seus próprios valores padrão.
        detector_kwargs = {}
        if args.model_path:
            detector_kwargs['model_path'] = args.model_path
        if args.conf_threshold:
            detector_kwargs['conf_threshold'] = args.conf_threshold

        # Criar instância do detector com os argumentos fornecidos
        detector = WheelDetector(**detector_kwargs)

        # Flag para exibir ou não a imagem depois
        show_flag = not args.no_show

        # Processar imagem SEM exibir (assim a função retorna imediatamente)
        results = detector.process_image(
            image_path=args.image_path,
            output_dir=args.save_path,
            show=False
        )

        # Imprimir resultados no console
        print("\n--- Resultados da Análise ---")
        print(f"  Imagem processada: {args.image_path}")
        print(f"  Resultado salvo em: {results['output_path']}")
        print(f"  Total de eixos detectados: {results['total_axles']}")
        print(f"  Eixos suspensos: {results['lifted_axles']}")
        for axle in results.get('axle_details', []):
            print(
                f"    - Eixo {axle['axle_number']}: "
                f"Confiança {axle['confidence']*100:.1f}%, "
                f"{axle['status']}"
            )

        # Exibir a imagem anotada ANTES de escolher a categoria
        if show_flag:
            annotated_img = cv2.imread(results['output_path'])
            if annotated_img is None:
                logger.warning("Não foi possível carregar a imagem anotada para exibição.")
            else:
                cv2.imshow('Detecção de Rodas e Eixos', annotated_img)
                print("\nAnálise visual: Pressione qualquer tecla na janela da imagem para continuar...")
                cv2.waitKey(0)
                cv2.destroyAllWindows()

        # Perguntar categoria de destino (a menos que --auto-categorize seja usado)
        selected_category = "output"  # valor padrão
        if not args.auto_categorize:
            selected_category = select_output_category()
            
            if selected_category != "output":
                # Mover arquivos para a categoria selecionada
                moved_files = move_results_to_category(
                    results['output_path'], 
                    results['json_results_path'], 
                    selected_category
                )
                print(f"\nArquivos movidos para: {selected_category}")
                print(f"Arquivos: {', '.join(moved_files)}")
            else:
                print(f"\nArquivos mantidos em: {results['output_path']}")
                print("(pasta padrão - não foi necessário mover arquivos)")

    except Exception as e:
        logger.error(f"Ocorreu um erro durante o processamento: {e}")
        raise

if __name__ == "__main__":
    main() 