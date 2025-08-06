import json
import os
import shutil
from pathlib import Path
from typing import Dict, List, Tuple

def analyze_result(json_path: str) -> Dict:
    """Analisa um arquivo de resultado JSON e retorna a categoria."""
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        total_axles = data.get('total_axles', 0)
        lifted_axles = data.get('lifted_axles', 0)
        total_wheels = data.get('total_wheels', 0)
        average_confidence = data.get('average_confidence', 0)
        axles = data.get('axles', [])
        
        # Análise das categorias
        categories = []
        
        # 1. Verificar se detectou todos os eixos esperados
        expected_axles = 3  # Assumindo que caminhões têm 3 eixos em média
        if total_axles < expected_axles:
            categories.append("Falha_Detecção_Eixos")
        
        # 2. Verificar se detectou eixos levantados corretamente
        if lifted_axles > 0:
            categories.append("Eixo_Levantado_Detectado")
        
        # 3. Verificar confiança média
        if average_confidence < 0.7:
            categories.append("Baixa_Confiança")
        elif average_confidence >= 0.8:
            categories.append("Alta_Confiança")
        
        # 4. Verificar se detectou todas as rodas
        expected_wheels = total_axles  # Assumindo 1 roda por eixo
        if total_wheels < expected_wheels:
            categories.append("Falha_Detecção_Rodas")
        
        # 5. Categoria principal baseada no sucesso geral
        if len(categories) == 0 or (len(categories) == 1 and "Eixo_Levantado_Detectado" in categories):
            categories.append("Detecção_Completa")
        
        return {
            'categories': categories,
            'total_axles': total_axles,
            'lifted_axles': lifted_axles,
            'total_wheels': total_wheels,
            'average_confidence': average_confidence
        }
        
    except Exception as e:
        print(f"Erro ao analisar {json_path}: {e}")
        return {'categories': ['Erro_Análise'], 'error': str(e)}

def organize_output_files(output_dir: str = "output"):
    """Organiza os arquivos de output em pastas por categoria."""
    output_path = Path(output_dir)
    
    if not output_path.exists():
        print(f"Diretório {output_dir} não encontrado!")
        return
    
    # Criar diretórios de categorias
    categories = [
        "01_Detecção_Completa",
        "02_Eixo_Levantado_Detectado", 
        "03_Falha_Detecção_Rodas",
        "04_Falha_Detecção_Eixos",
        "05_Baixa_Confiança",
        "06_Alta_Confiança",
        "07_Erro_Análise"
    ]
    
    for category in categories:
        category_path = output_path / category
        category_path.mkdir(exist_ok=True)
    
    # Processar arquivos JSON
    json_files = list(output_path.glob("*.json"))
    
    print(f"Encontrados {len(json_files)} arquivos de resultado")
    
    for json_file in json_files:
        print(f"\nAnalisando: {json_file.name}")
        
        # Analisar resultado
        analysis = analyze_result(str(json_file))
        print(f"  Categorias: {analysis['categories']}")
        print(f"  Eixos: {analysis.get('total_axles', 'N/A')}")
        print(f"  Eixos levantados: {analysis.get('lifted_axles', 'N/A')}")
        print(f"  Rodas: {analysis.get('total_wheels', 'N/A')}")
        print(f"  Confiança média: {analysis.get('average_confidence', 'N/A'):.3f}")
        
        # Determinar categoria principal
        primary_category = analysis['categories'][0] if analysis['categories'] else "07_Erro_Análise"
        
        # Mover arquivos relacionados
        base_name = json_file.stem
        jpg_file = json_file.with_suffix('.jpg')
        
        target_dir = output_path / primary_category
        
        # Mover JSON
        if json_file.exists():
            shutil.move(str(json_file), str(target_dir / json_file.name))
            print(f"  Movido JSON para: {primary_category}")
        
        # Mover JPG se existir
        if jpg_file.exists():
            shutil.move(str(jpg_file), str(target_dir / jpg_file.name))
            print(f"  Movido JPG para: {primary_category}")
    
    # Mostrar resumo
    print("\n" + "="*50)
    print("RESUMO DA ORGANIZAÇÃO:")
    print("="*50)
    
    for category in categories:
        category_path = output_path / category
        if category_path.exists():
            files = list(category_path.glob("*"))
            if files:
                print(f"{category}: {len(files)} arquivos")
    
    print("\nOrganização concluída!")

if __name__ == "__main__":
    organize_output_files() 