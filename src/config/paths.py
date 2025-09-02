"""
Configuração de caminhos do sistema.
"""
from pathlib import Path

# Diretório raiz do projeto (2 níveis acima de src/)
PROJECT_ROOT = Path(__file__).parent.parent.parent

# Modelos treinados
MODELS_DIR = PROJECT_ROOT / "wheel-detector-v11n-finetuned"
WHEEL_MODEL_PATH = MODELS_DIR / "weights" / "best.pt"

# Diretórios de dados
IMAGES_DIR = PROJECT_ROOT / "images"
HIGHWAY_DATASET_DIR = PROJECT_ROOT / "highway_dataset"
TEST_CAMERA_01_DIR = PROJECT_ROOT / "testes-camera-01"
TEST_CAMERA_02_DIR = PROJECT_ROOT / "testes-camera-02"

# Diretórios de saída
OUTPUT_DIR = PROJECT_ROOT / "output"
VIDEO_OUTPUT_DIR = PROJECT_ROOT / "video_output"

# Diretório de ambiente virtual
VENV_DIR = PROJECT_ROOT / "venv"

def ensure_directories():
    """Cria diretórios necessários se não existirem."""
    directories = [
        OUTPUT_DIR,
        VIDEO_OUTPUT_DIR
    ]
    
    for directory in directories:
        directory.mkdir(exist_ok=True)
        print(f"Diretório {directory} verificado/criado com sucesso!")

def get_relative_path(path: Path) -> str:
    """Retorna o caminho relativo ao diretório raiz do projeto."""
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)
