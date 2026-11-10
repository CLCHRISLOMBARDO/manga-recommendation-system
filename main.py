import os
import pandas as pd
import numpy as np

# Importaciones de tus módulos locales
from src.config import  TARGET
from src.utils import folders_creation,train_test_split_dfs,evaluar_experimentos      
from src.feat_eng import pipeline_feature_engineering
from src.filtrado_colaborativo import filtrado_colaborativo
from src.construccion_cache import construir_cache
from src.entrenar_ease import obtener_modelo_ease
import src.retrievals as ret
import src.rankers as rnk


def main():
    print("=" * 60)
    print("INICIANDO PIPELINE DE SISTEMA DE RECOMENDACIÓN DE LIBROS")
    print("=" * 60)

    #if state =='entrenamiento': 
    print('*'*30+f"INICIANDO CON STATE : {'entrenamiento'}")
    # 1. Asegurar que existan los directorios de outputs necesarios
    folders_creation()
    #os.makedirs("outputs/modelo", exist_ok=True)

    # 2. Split de Train y Test (Basado en los últimos 20 libros por usuario)
    print("\n--- PASO 1: Generando / Cargando splits de Train y Test ---")
    #df_train, df_test= train_test_split_dfs()
    df_train, df_test = train_test_split_dfs()
    print(f"Filas en Train: {len(df_train)} | Filas en Test: {len(df_test)}")
    
    
    

    # 3. Feature Engineering sobre Train
    # Esto calcula métricas históricas, guarda el archivo en disco y devuelve el df enriquecido
    print("\n--- PASO 2: Ejecutando Feature Engineering en Train ---")
    df_train_features = pipeline_feature_engineering(df_train, state='entrenamiento')

    # Limpieza de seguridad por si quedaron nulos en el target
    df_train_features = df_train_features.dropna(subset=[TARGET])


    
    # 6. FILTRADO COLABORATIVO
    m_lectores, m_libros = filtrado_colaborativo(state='entrenamiento')
    
    # 9. EASE
    modelo_ease = obtener_modelo_ease(df_train, state="entrenamiento")
    
    # 7. Construccion del caché
    cache = construir_cache(df_train_features ,modelo_lgbm_reg=None, modelo_lgbm_ranker=None,modelo_ease=modelo_ease, state='entrenamiento')
    
    # 8. Definicion estrategia evaluacion
    estrategias_retrieval = {
        "Q1_Populares": ret.retrieval_q1_populares,
        "Q2_Mismo_Autor": ret.retrieval_q2_mismo_autor,
        "Q3_Mismo_Genero": ret.retrieval_q3_mismo_genero,
        "Q4_Popularidad_Pais": ret.retrieval_q4_popularidad_pais,
        "Q5_Trending": ret.retrieval_q5_trending,
        "Hibrido": ret.retrieval_hibrido,
        "Q6_Productos_Asociados": ret.retrieval_q6_productos_asociados,
        "Q7_EASE": ret.retrieval_q7_ease
    }
    estrategias_ranker = {
        "Popularidad": rnk.ranker_popularidad,
        "Jaccard": rnk.ranker_jaccard,
        "CF_DotProduct": rnk.ranker_filtrado_colaborativo_dot_product,
        "CF_Promedio": rnk.ranker_filtrado_colaborativo_promedio_scores,
        "EASE_Ranker": rnk.ranker_ease
        #"LGBM_Regressor": rnk.ranker_lgbm_regressor,
       #"LGBM_Ranker": rnk.ranker_lgbm_ranker
    }

    # 9. Evaluación de Experimentos (Retrievals x Rankers)
    print("\n---  Evaluando matriz de experimentos (Retrievals + Ranking) ---")
    df_resultados = evaluar_experimentos(
        df_test=df_test,
        cache=cache,
        estrategias_retrieval=estrategias_retrieval,
        estrategias_ranker=estrategias_ranker,
        top_n_retrieval=100
    )
    

    print("\n" + "=" * 60)
    print("RESULTADOS FINALES DE LOS EXPERIMENTOS:")
    print("=" * 60)
    print(df_resultados.to_string(index=False))

    print('*'*30 +"FIN DEL PROCESO" + '*'*30 )



if __name__ == "__main__":
    main()