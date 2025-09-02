"""
Configuração de caminhos do sistema.
"""
from pathlib import Path
from typing import Dict, List
import os

class PathManager:
    """Gerenciador centralizado de caminhos do sistema."""
    
    def __init__(self):
        # Diretório raiz do projeto (3 níveis acima de src/infrastructure/config/)
        self.project_root = Path(__file__).parent.parent.parent.parent
        
        # Modelos treinados
        self.models_dir = self.project_root / "wheel-detector-v11n-finetuned"
        self.wheel_model_path = self.models_dir / "weights" / "best.pt"
        
        # Diretórios de dados
        self.images_dir = self.project_root / "images"
        self.highway_dataset_dir = self.project_root / "highway_dataset"
        self.test_camera_01_dir = self.project_root / "testes-camera-01"
        self.test_camera_02_dir = self.project_root / "testes-camera-02"
        
        # Diretórios de saída
        self.output_dir = self.project_root / "output"
        self.video_output_dir = self.project_root / "video_output"
        
        # Diretório de ambiente virtual
        self.venv_dir = self.project_root / "venv"
        
        # Diretórios temporários
        self.temp_dir = self.project_root / "temp"
        self.cache_dir = self.project_root / "cache"
        
        # Diretórios de logs
        self.logs_dir = self.project_root / "logs"
    
    def ensure_directories(self) -> None:
        """Cria diretórios necessários se não existirem."""
        directories = [
            self.output_dir,
            self.video_output_dir,
            self.temp_dir,
            self.cache_dir,
            self.logs_dir
        ]
        
        for directory in directories:
            directory.mkdir(exist_ok=True)
            print(f"Diretório {directory} verificado/criado com sucesso!")
    
    def get_relative_path(self, path: Path) -> str:
        """Retorna o caminho relativo ao diretório raiz do projeto."""
        try:
            return str(path.relative_to(self.project_root))
        except ValueError:
            return str(path)
    
    def get_model_paths(self) -> Dict[str, Path]:
        """Retorna todos os caminhos de modelos."""
        return {
            "wheel_model": self.wheel_model_path,
            "models_dir": self.models_dir
        }
    
    def get_data_paths(self) -> Dict[str, Path]:
        """Retorna todos os caminhos de dados."""
        return {
            "images": self.images_dir,
            "highway_dataset": self.highway_dataset_dir,
            "test_camera_01": self.test_camera_01_dir,
            "test_camera_02": self.test_camera_02_dir
        }
    
    def get_output_paths(self) -> Dict[str, Path]:
        """Retorna todos os caminhos de saída."""
        return {
            "output": self.output_dir,
            "video_output": self.video_output_dir,
            "temp": self.temp_dir,
            "cache": self.cache_dir,
            "logs": self.logs_dir
        }
    
    def validate_paths(self) -> List[str]:
        """Valida se todos os caminhos críticos existem."""
        errors = []
        
        # Verificar modelos
        if not self.wheel_model_path.exists():
            errors.append(f"Modelo de roda não encontrado: {self.wheel_model_path}")
        
        # Verificar diretórios de dados
        if not self.images_dir.exists():
            errors.append(f"Diretório de imagens não encontrado: {self.images_dir}")
        
        return errors

# Instância global do gerenciador de caminhos
path_manager = PathManager()

# Aliases para compatibilidade com código existente
PROJECT_ROOT = path_manager.project_root
MODELS_DIR = path_manager.models_dir
WHEEL_MODEL_PATH = path_manager.wheel_model_path
IMAGES_DIR = path_manager.images_dir
HIGHWAY_DATASET_DIR = path_manager.highway_dataset_dir
TEST_CAMERA_01_DIR = path_manager.test_camera_01_dir
TEST_CAMERA_02_DIR = path_manager.test_camera_02_dir
OUTPUT_DIR = path_manager.output_dir
VIDEO_OUTPUT_DIR = path_manager.video_output_dir
VENV_DIR = path_manager.venv_dir

# Função de compatibilidade
def ensure_directories():
    """Função de compatibilidade para código existente."""
    path_manager.ensure_directories()
