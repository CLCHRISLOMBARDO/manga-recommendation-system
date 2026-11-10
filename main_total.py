import os
import pandas as pd
import numpy as np

# Importaciones de tus módulos locales
from src.config import OUTPUTS_DATASETS, TARGET, FEATURES, FEATURES_OUT
from src.utils import folders_creation, cracion_df_total,model_baseline_train_lgbm_reg ,model_baseline_train_lgbm_ranker, calcular_feature_importance,evaluar_experimentos      
from src.feat_eng import pipeline_feature_engineering
from src.filtrado_colaborativo import filtrado_colaborativo
from src.construccion_cache import construir_cache
from src.entrenar_ease import obtener_modelo_ease
import src.retrievals as ret
import src.rankers as rnk
import pickle


def main():
    print("=" * 60)
    print("INICIANDO PIPELINE DE SISTEMA DE RECOMENDACIÓN DE LIBROS")
    print("=" * 60)

    #if state =='entrenamiento': 
    print('*'*30+f"INICIANDO CON STATE : {'entrenamiento final'}")
    # 1. Asegurar que existan los directorios de outputs necesarios
   

    # 2. Split de Train y Test (Basado en los últimos 20 libros por usuario)
    print("\n--- PASO 1: Generando df_total ---")
    df_total = cracion_df_total()

    
        

    # 3. Feature Engineering sobre Train
    # Esto calcula métricas históricas, guarda el archivo en disco y devuelve el df enriquecido
    print("\n--- PASO 2: Ejecutando Feature Engineering en Train ---")
    df_total_features = pipeline_feature_engineering(df_total, state='entrenamiento_final')


 
    
    # 6. FILTRADO COLABORATIVO
    m_lectores, m_libros = filtrado_colaborativo(state='entrenamiento_final')
    
    # 8. EntreNAS ease
    modelo_ease = obtener_modelo_ease(df_total, state="entrenamiento_final")
    
    # 7. Construccion del caché
    cache = construir_cache(df_total_features ,modelo_lgbm_reg=None, modelo_lgbm_ranker=None,modelo_ease=modelo_ease, state='entrenamiento_final')
    
    with open('outputs/cache_final/cache.pkl', 'wb') as f:
        pickle.dump(cache, f)


    print('*'*30 +"FIN DEL PROCESO" + '*'*30 )



if __name__ == "__main__":
    main()