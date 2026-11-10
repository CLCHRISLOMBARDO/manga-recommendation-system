import pandas as pd
import numpy as np
import sqlite3
import os

from sklearn.model_selection import train_test_split
from lightgbm import LGBMRegressor ,LGBMRanker
import joblib
import json
import time



from src.config import BASE_DB,TABLA_MODELO_FULL ,OUTPUTS_MODELO_TRAIN, OUTPUTS_MODELO_FINAL,OUTPUTS_DATASETS ,OUTPUTS_CHECKPOINTS ,OUTPUTS_RESULTADOS
from src.config import FEATURES ,CAT_FEATURES,TARGET , OUTPUTS_EASE,OUTPUTS_FILTRADO_COLAB
from src.metrics import calcular_recall, calcular_ndcg,calcular_ndcg_estricto  
# Creacion de carpetas
def folders_creation():
    
    os.makedirs(OUTPUTS_DATASETS, exist_ok=True)
    os.makedirs(OUTPUTS_MODELO_TRAIN, exist_ok=True)
    os.makedirs(OUTPUTS_MODELO_FINAL, exist_ok=True)
    os.makedirs(OUTPUTS_CHECKPOINTS, exist_ok=True)
    os.makedirs(OUTPUTS_RESULTADOS, exist_ok=True)
    os.makedirs(OUTPUTS_FILTRADO_COLAB, exist_ok=True)
    os.makedirs(OUTPUTS_EASE, exist_ok=True)

# Split train test

def cracion_df_total(tabla=TABLA_MODELO_FULL):
    total_file = "df_total.parquet"
    exist_total = False
    if os.path.exists(OUTPUTS_DATASETS + total_file):
        print("Ya existe archivo total")
        exist_total = True

    if exist_total  :
        print(f" exist_total :{exist_total} ")
        # 2. Usar read_parquet
        df_total = pd.read_parquet(OUTPUTS_DATASETS + total_file)
        return df_total
    else: 
        con = sqlite3.connect(BASE_DB)
        df_total = pd.read_sql_query(f"""select id_lector, id_libro, fecha_interaccion, rating FROM {tabla}""",con)

        df_total.to_parquet(OUTPUTS_DATASETS + total_file)
    return df_total

    
def train_test_split_dfs(tabla = TABLA_MODELO_FULL): 
    print(f"************************************ Comenzando split train test ************************************")
    # 1. Cambiar ambas extensiones a .parquet
    train_file = "df_train.parquet"
    test_file = "df_test.parquet"
    
    exist_train = False
    exist_test = False 
    if os.path.exists(OUTPUTS_DATASETS + train_file):
        print("Ya existe archivo train")
        exist_train = True
    if os.path.exists(OUTPUTS_DATASETS + test_file):
        print("Ya existe archivo test")
        exist_test = True

    if exist_train and exist_test :
        print(f"exist train : {exist_train}   and   exist_test :{exist_test}  ")
        # 2. Usar read_parquet
        df_train = pd.read_parquet(OUTPUTS_DATASETS + train_file)
        df_test = pd.read_parquet(OUTPUTS_DATASETS + test_file)
        return df_train , df_test
    else: 
        con = sqlite3.connect(BASE_DB)
        id_lectores = pd.read_sql(f"""
                        SELECT id_lector
                        FROM {tabla}
                        GROUP BY id_lector
                        HAVING COUNT(*) >= 10
                    """, con)["id_lector"]
        df_train = pd.DataFrame({"id_lector": [], "id_libro": [], "rating": []})
        df_test = pd.DataFrame({"id_lector": [], "id_libro": [], "rating": []})

        for id_lector in id_lectores:
            id_libros = pd.read_sql(f"""
                SELECT id_lector, id_libro, fecha_interaccion, rating
                FROM {tabla}
                WHERE id_lector = ?
                ORDER BY fecha_interaccion
            """, con, params=[id_lector])

            df_train = pd.concat([df_train, id_libros[:-10]], axis=0)
            df_test = pd.concat([df_test, id_libros[-10:]], axis=0)        
        # 3. Usar to_parquet (index=False se omite porque Parquet maneja el índice automáticamente)
        df_train.to_parquet(OUTPUTS_DATASETS + train_file)
        df_test.to_parquet(OUTPUTS_DATASETS + test_file)
        
        df_train.to_sql(name='interacciones_train', con=con, if_exists='replace', index=False)
        df_test.to_sql(name='interacciones_test', con=con, if_exists='replace', index=False)
        con.close()
    return df_train , df_test 


def train_test_split_dfs_corte_year(tabla = TABLA_MODELO_FULL,year=2024): 
    print(f"************************************ Comenzando split train test ************************************")
    # 1. Cambiar ambas extensiones a .parquet
    train_file = "df_train.parquet"
    test_file = "df_test.parquet"

    exist_train = False
    exist_test = False 
    if os.path.exists(OUTPUTS_DATASETS + train_file):
        print("Ya existe archivo train")
        exist_train = True
    if os.path.exists(OUTPUTS_DATASETS + test_file):
        print("Ya existe archivo test")
        exist_test = True


    if exist_train and exist_test:
        print(f"exist train : {exist_train}   and   exist_test :{exist_test}  ")
        # 2. Usar read_parquet
        df_train = pd.read_parquet(OUTPUTS_DATASETS + train_file)
        df_test = pd.read_parquet(OUTPUTS_DATASETS + test_file)
        return df_train , df_test 
    else: 
        
        
        df_total = cracion_df_total()
        
        df_total['col_fecha'] = pd.to_datetime(df_total['fecha_interaccion'],format='%Y-%m-%d' ,errors='coerce')
        df_total['year'] = df_total['col_fecha'].dt.year.astype('Int64')
        df_train = df_total.copy()
        df_train = df_train[df_train['year']<year]
        df_test = df_total.copy()
        df_test = df_test[df_test['year'] >= year]
        
        print("*"*30+" VALORES df_total:")
        print(f"Shape:{df_total.shape} \nLectores unicos: {df_total.id_lector.nunique()} \nAños:{df_total.year.value_counts(dropna=False).sort_index()} ")
        print(f"Shape:{df_train.shape} \nLectores unicos: {df_train.id_lector.nunique()} \nAños:{df_train.year.value_counts(dropna=False).sort_index()} ")
        print(f"Shape:{df_test.shape} \nLectores unicos: {df_test.id_lector.nunique()} \nAños:{df_test.year.value_counts(dropna=False).sort_index()} ")
        
        # 3. Usar to_parquet (index=False se omite porque Parquet maneja el índice automáticamente)
        df_train.to_parquet(OUTPUTS_DATASETS + train_file)
        df_test.to_parquet(OUTPUTS_DATASETS + test_file)
        
        con = sqlite3.connect(BASE_DB)
        df_train.to_sql(name='interacciones_train', con=con, if_exists='replace', index=False)
        df_test.to_sql(name='interacciones_test', con=con, if_exists='replace', index=False)
        con.close()
        df_train = df_train.drop(columns = ['col_fecha','year'])
        df_test = df_test.drop(columns = ['col_fecha','year'])
    return df_train , df_test 

# MODELOS LGBM
def model_baseline_train_lgbm_reg(df_train:pd.DataFrame ,features_seleccionadas:list[str] , seed:int):
    '''df_train puede ser solo el de la fase de entrenamiento o el total incluyendo al test para la pred final'''
    print("************************************ Comienzo entrenamiendo modelo base ************************************")
    print(f"Features seleccionadas para el entrenamiento : {features_seleccionadas}")
    print(f"Target : {TARGET}")

    cat_features=[c for c in CAT_FEATURES if c in features_seleccionadas]

    for c in cat_features:
            df_train[c] = df_train[c].astype('category')

    modelo_lgbm = LGBMRegressor(
        random_state=seed
    )
    X_train = df_train[features_seleccionadas]
    y_train = df_train[TARGET]

    modelo_lgbm.fit(
        X_train, 
        y_train,
        categorical_feature=cat_features 
    ) 
    joblib.dump(modelo_lgbm, OUTPUTS_MODELO_TRAIN +f'modelo_lgbm_reg.pkl')
    return modelo_lgbm
def model_baseline_train_lgbm_ranker(df_train: pd.DataFrame, features_seleccionadas: list[str], seed: int):
    print("************************************ Comienzo entrenamiento modelo RANKER (LGBMRanker) ************************************")
    print(f"Features seleccionadas para el entrenamiento : {features_seleccionadas}")
    print(f"Target : {TARGET}")

    # 1. ORDENAR POR USUARIO (CRÍTICO PARA RANKING)
    # LightGBM exige que todas las interacciones de un mismo query (usuario) estén juntas contiguamente.
    df_train = df_train.sort_values(by="id_lector").reset_index(drop=True)

    # 2. ARMAR EL ARREGLO DE GRUPOS
    # Contamos exactamente cuántos libros tiene cada usuario en este set de entrenamiento.
    # sort=False es vital porque ya ordenamos el DataFrame arriba.
    group_counts = df_train.groupby("id_lector", sort=False).size().values

    cat_features = [c for c in CAT_FEATURES if c in features_seleccionadas]

    for c in cat_features:
        df_train[c] = df_train[c].astype('category')

    # 3. INICIALIZAR EL MODELO RANKER
    # lambdarank es la función de pérdida matemática estándar para optimizar NDCG
    modelo_lgbm = LGBMRanker(
        objective="lambdarank",
        metric="ndcg",
        random_state=seed,
        importance_type="gain"
    )
    
    X_train = df_train[features_seleccionadas]
    y_train = df_train[TARGET]

    # 4. ENTRENAR PASANDO EL PARÁMETRO GROUP
    modelo_lgbm.fit(
        X_train, 
        y_train,
        group=group_counts,
        categorical_feature=cat_features 
    ) 
    joblib.dump(modelo_lgbm, OUTPUTS_MODELO_TRAIN +f'modelo_lgbm_ranker.pkl')
    
    return modelo_lgbm
def calcular_feature_importance(modelo_lgbm, model_type :str ) -> pd.DataFrame:
    """
    Extrae la importancia de las variables de un modelo LightGBM entrenado,
    calculando por Split, por Gain y el porcentaje relativo del Gain.
    """
    print("************************************ Calculando Feature Importances ************************************")
    
    # 1. Extraemos los nombres de las variables directamente del modelo
    features = modelo_lgbm.booster_.feature_name()
    
    # 2. Extraemos ambas métricas desde el 'booster' interno de LightGBM
    importancia_split = modelo_lgbm.booster_.feature_importance(importance_type='split')
    importancia_gain = modelo_lgbm.booster_.feature_importance(importance_type='gain')
    
    # 3. Armamos el DataFrame
    df_importances = pd.DataFrame({
        'Feature': features,
        'Split': importancia_split,
        'Gain': importancia_gain
    })
    
    # 4. Calculamos el porcentaje sobre la ganancia total (Gain)
    total_gain = df_importances['Gain'].sum()
    df_importances['Gain_Porcentaje'] = (df_importances['Gain'] / total_gain) * 100
    
    # 5. Ordenamos por el porcentaje de Gain (de mayor a menor)
    df_importances = df_importances.sort_values(by='Gain_Porcentaje', ascending=False).reset_index(drop=True)
    
    file_name = OUTPUTS_MODELO_TRAIN + f"feat_imp_{model_type}.csv"
    df_importances.to_csv(file_name)
    print(f"feat importances guardado en : {file_name}")
    
    return df_importances

# EVALUAR EXPERIMENTOS

def evaluar_experimentos(df_test, cache, estrategias_retrieval, estrategias_ranker, top_n_retrieval=100):
    """
    Motor agnóstico de evaluación. Cruza cualquier Retrieval con cualquier Ranker.
    """
    path_checkpoints = OUTPUTS_CHECKPOINTS
    path_resultados = OUTPUTS_RESULTADOS
    
    resultados_totales = []
    usuarios_test = df_test["id_lector"].unique()
    
    # 1. OPTIMIZACIÓN: Pre-calculamos el ground truth de Test para todos los usuarios de una vez
    # Formato: {id_lector: {id_libro: rating}}
    print("Pre-calculando Ground Truth de Test...")
    ground_truth = (
        df_test.groupby("id_lector")
        .apply(lambda x: dict(zip(x["id_libro"], x["rating"])))
        .to_dict()
    )

    # 2. Bucles de Evaluación
    for nombre_retrieval, func_retrieval in estrategias_retrieval.items():
        for nombre_ranker, func_ranker in estrategias_ranker.items():
            
            print(f"\n{'='*70}")
            print(f"Evaluando: Retrieval=[{nombre_retrieval}] | Ranker=[{nombre_ranker}]")
            print(f"{'='*70}")
            
            inicio = time.time()
            checkpoint_file = os.path.join(path_checkpoints, f"ckpt_{nombre_retrieval}_{nombre_ranker}.csv")
            
            usuarios_procesados = set()
            ndcg_lista = []
            recall_lista = []
            
            # --- RECUPERA CHECKPOINT ---
            if os.path.exists(checkpoint_file):
                print(f" -> Encontrado checkpoint previo: {checkpoint_file}. Retomando...")
                df_ckpt = pd.read_csv(checkpoint_file)
                for _, row in df_ckpt.iterrows():
                    usuarios_procesados.add(row["id_lector"])
                    recall_lista.append(row["recall"])
                    ndcg_lista.append(row["ndcg"])
                    
            nuevos_procesados = 0
            
            # --- BUCLE POR USUARIO ---
            for id_lector in usuarios_test:
                if id_lector in usuarios_procesados:
                    continue
                    
                # Obtenemos los libros que realmente leyó y sus ratings
                true_relevance_dict = ground_truth.get(id_lector, {})
                
                # --- FASE 1: RETRIEVAL ---
                candidatos = func_retrieval(id_lector, top_n=top_n_retrieval, cache=cache)
                recall = calcular_recall(candidatos, true_relevance_dict)
                
                # --- FASE 2: RANKING ---
                scores_predichos = func_ranker(id_lector, candidatos, cache=cache)
                ndcg = calcular_ndcg_estricto(true_relevance_dict, scores_predichos, k=10)
                
                # Guardamos en memoria
                recall_lista.append(recall)
                ndcg_lista.append(ndcg)
                
                # --- GUARDA CHECKPOINT ---
                registro_nuevo = pd.DataFrame([{
                    "id_lector": id_lector,
                    "recall": recall,
                    "ndcg": ndcg
                }])
                registro_nuevo.to_csv(checkpoint_file, mode='a', header=not os.path.exists(checkpoint_file), index=False)
                
                nuevos_procesados += 1
                if nuevos_procesados % 100 == 0:
                    print(f"   -> Procesados {nuevos_procesados} usuarios nuevos...")
            
            # --- FIN DEL EXPERIMENTO (Cálculo de Promedios) ---
            tiempo_total = round(time.time() - inicio, 2)
            mean_recall = np.mean(recall_lista) if recall_lista else 0.0
            mean_ndcg = np.mean(ndcg_lista) if ndcg_lista else 0.0
            
            resultados_totales.append({
                "Retrieval": nombre_retrieval,
                "Ranker": nombre_ranker,
                "Recall_Retrieval_Promedio": mean_recall,
                "NDCG@20_Promedio": mean_ndcg,
                "Tiempo_Segundos": tiempo_total
            })
            
            print(f" -> RESULTADO | NDCG@20: {mean_ndcg:.4f} | Recall: {mean_recall:.4f} | Tiempo: {tiempo_total}s")
            
    # --- EXPORTACIÓN FINAL ---
    df_resultados = pd.DataFrame(resultados_totales)
    os.makedirs(path_resultados, exist_ok=True)
    archivo_final = os.path.join(path_resultados, "resultados_experimentos.csv")
    df_resultados.to_csv(archivo_final, index=False)
    
    print(f"\nExperimentos finalizados. Resultados guardados en: {archivo_final}")
    return df_resultados

