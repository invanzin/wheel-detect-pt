"""
Arquivo principal do sistema de detecção de rodas.
Este é um arquivo temporário para demonstrar a nova estrutura.
"""
import logging
from pathlib import Path

# Importar configurações
from infrastructure.config.settings import settings
from infrastructure.config.paths import path_manager

# Configurar logging
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL),
    format=settings.LOG_FORMAT
)
logger = logging.getLogger(__name__)

def main():
    """Função principal do sistema."""
    logger.info("Iniciando sistema de detecção de rodas...")
    
    try:
        # Validar e criar diretórios
        path_manager.ensure_directories()
        
        # Validar caminhos críticos
        errors = path_manager.validate_paths()
        if errors:
            logger.error("Erros de validação encontrados:")
            for error in errors:
                logger.error(f"  - {error}")
            return
        
        # Mostrar configurações
        logger.info("Configurações do sistema:")
        logger.info(f"  - Threshold de confiança: {settings.ML_CONFIDENCE_THRESHOLD}")
        logger.info(f"  - Tamanho da imagem: {settings.ML_IMAGE_SIZE}")
        logger.info(f"  - Diretório de saída: {path_manager.output_dir}")
        
        logger.info("Sistema iniciado com sucesso!")
        
    except Exception as e:
        logger.error(f"Erro ao iniciar sistema: {e}")
        raise

if __name__ == "__main__":
    main()
