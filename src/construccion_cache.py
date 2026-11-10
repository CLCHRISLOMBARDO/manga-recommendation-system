import pandas as pd
import sqlite3

from src.config import BASE_DB,OUTPUTS_FILTRADO_COLAB

def construir_cache(df ,modelo_lgbm_reg, modelo_lgbm_ranker,modelo_ease, state='entrenamiento' ):
    '''
    df puede ser df completo (df_total_features) con state=entrenamiento o df_train (df_train_features) pero con las features calculadas. 
    '''
    
    print("Construyendo Caché para Retrievals...")
    cache = {}
    
    con = sqlite3.connect(BASE_DB)
    
    # 0. Guardamos el df con todas las features 
    cache["df_train_features"] = df
    
    # 1. Traemos los catálogos por separado
    df_libros = pd.read_sql_query("""
        SELECT id_libro, titulo, autor, genero as genero_libro, 
                CAST(anio_edicion AS integer) AS anio_edicion 
        FROM libros
    """, con)
    
    # --- NUEVO PARA Q6 (Jaccard) ---
    
    # A. Diccionario de ratings por usuario {id_lector: {id_libro: rating}}
    cache["dict_leidos_ratings"] = (
        df.groupby('id_lector')
        .apply(lambda x: dict(zip(x['id_libro'], x['rating'])))
        .to_dict()
    )
    
    # B. Traemos la tabla de Jaccard de SQL
    if state=='entrenamiento':
        tabla_sql_productos_asociados = "PRODUCTOS_ASOCIADOS_TRAIN"
    elif state =='entrenamiento_final':
        tabla_sql_productos_asociados = "PRODUCTOS_ASOCIADOS"
        
    df_jaccard = pd.read_sql_query(f"SELECT libro_A, libro_B, jaccard_score FROM {tabla_sql_productos_asociados}", con)
    

    con.close()
    
    # 1. Libros leídos por usuario {id_lector: set(libros_leidos)}
    cache["dict_leidos"] = df.groupby("id_lector")["id_libro"].apply(set).to_dict()
    
    # 2. Populares globales (Q1)
    cache["populares_global"] = (
        df.groupby("id_libro")["rating_promedio_te_por_libro"].max()
        .sort_values(ascending=False).index.tolist()
    )
    
    # 3. Mapas para Q2 y Q3 (Libro -> Autor/Género y Autor/Género -> Libros)
    cache["dict_libro_autor"] = df_libros.set_index("id_libro")["autor"].to_dict()
    cache["dict_libro_genero"] = df_libros.set_index("id_libro")["genero_libro"].to_dict() 
    
    # Agrupamos libros por autor y por género
    cache["libros_por_autor"] = df_libros.groupby("autor")["id_libro"].apply(set).to_dict()
    cache["libros_por_genero"] = df_libros.groupby("genero_libro")["id_libro"].apply(set).to_dict()
    
    # 4. Populares por País (Q4)
    cache["dict_lector_pais"] = df.drop_duplicates(subset='id_lector').set_index("id_lector")["vive_en"].to_dict()
    
    # Unimos train con lectores para tener el país de cada lectura
    
    cache["populares_por_pais"] = {}
    for pais, df_sub in df.groupby("vive_en"):
        cache["populares_por_pais"][pais] = (
           df_sub.groupby("id_libro")["rating_promedio_te_por_libro"].max()
            .sort_values(ascending=False).index.tolist()
        )
        
    # 5. Populares Trending (Q5) - Últimos 12 meses
    if "fecha_interaccion" in df.columns:
        fechas_dt = pd.to_datetime(df["fecha_interaccion"], format="%Y-%m-%d", errors='coerce')
        fecha_corte = fechas_dt.max() - pd.DateOffset(months=12)
        df_reciente = df[fechas_dt >= fecha_corte]
        
        cache["populares_trending"] = (
            df_reciente.groupby("id_libro")["rating_promedio_te_por_libro"].max()
            .sort_values(ascending=False).index.tolist()
        )
    else:
        print("WARNING: no estaba la fecha_interaccion en df")
        cache["populares_trending"] = cache["populares_global"] # Fallback
        
    cache["dict_popularidad_score"] = df.groupby("id_libro")["rating_promedio_te_por_libro"].max().to_dict()
    cache["media_global"] = df["rating"].mean()
    
    # 6. PRODUCTOS ASOCIADOS - JACCARD
    # Formato: {libro_X: {libro_Y: score, libro_Z: score}}
    jaccard_matrix = {}
    # Usar zip es 100 veces más rápido que iterrows()
    for a, b, score in zip(df_jaccard['libro_A'], df_jaccard['libro_B'], df_jaccard['jaccard_score']):
        if a not in jaccard_matrix: jaccard_matrix[a] = {}
        if b not in jaccard_matrix: jaccard_matrix[b] = {}
        jaccard_matrix[a][b] = score
        jaccard_matrix[b][a] = score
        
    cache["jaccard_matrix"] = jaccard_matrix
    
    # 7. FILTRADO COLABORATIVO
    m_lectores = pd.read_parquet(OUTPUTS_FILTRADO_COLAB + f'matriz_comportamiento_perfiles_lectores_{state}.parquet')
    m_libros = pd.read_parquet(OUTPUTS_FILTRADO_COLAB + f'matriz_comportamiento_perfiles_libros_{state}.parquet')
    cache["matriz_lectores"] = m_lectores
    cache["matriz_libros"] = m_libros
    
    # 8. a) MODELOS LGBM MACHINE LEARNING - REGRESOR
    if modelo_lgbm_reg is not None:
        cache["modelo_lgbm_reg"] = modelo_lgbm_reg
        cache["features_lgbm_reg"] = modelo_lgbm_reg.booster_.feature_name()
    
    # 8. b) MODELOS LGBM MACHINE LEARNING - RANKER
    if modelo_lgbm_ranker is not None:
        cache["modelo_lgbm_ranker"] = modelo_lgbm_ranker
        cache["features_lgbm_ranker"] = modelo_lgbm_ranker.booster_.feature_name()
    
    if modelo_ease is not None:
        cache["modelo_ease"] = modelo_ease
        
    print(f"Caché  de {state} construido con éxito.")
    return cache
