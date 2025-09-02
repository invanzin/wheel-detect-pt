"""
Utilitários para manipulação de arquivos.
"""
import os
import shutil
from pathlib import Path
from typing import List, Dict
import json

def create_directory(path: Path) -> None:
    """Cria um diretório se não existir."""
    path.mkdir(parents=True, exist_ok=True)

def get_files_by_extension(directory: Path, extensions: List[str]) -> List[Path]:
    """Retorna arquivos com extensões específicas de um diretório."""
    files = []
    for ext in extensions:
        files.extend(directory.glob(f"*.{ext}"))
    return sorted(files)

def copy_file_with_timestamp(source: Path, destination_dir: Path, suffix: str = "") -> Path:
    """Copia um arquivo para um diretório com timestamp."""
    from datetime import datetime
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    new_name = f"{source.stem}_{timestamp}{suffix}{source.suffix}"
    destination = destination_dir / new_name
    
    shutil.copy2(source, destination)
    return destination

def save_json_results(data: Dict, file_path: Path) -> None:
    """Salva resultados em formato JSON."""
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def load_json_results(file_path: Path) -> Dict:
    """Carrega resultados de um arquivo JSON."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)
