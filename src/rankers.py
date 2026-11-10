import pandas as pd
from src.feat_eng import pipeline_feature_engineering
from src.config import CAT_FEATURES

def ranker_jaccard(id_lector, candidatos, cache):
    leidos_ratings = cache["dict_leidos_ratings"].get(id_lector, {})
    scores = {}
    
    for candidato in candidatos:
        score_final = 0.0
        # Vemos a qué libros está conectado este candidato
        conexiones = cache["jaccard_matrix"].get(candidato, {})
        
        # Si el usuario leyó alguno de esos libros conectados, sumamos (Rating * Jaccard)
        for lib_leido, rating in leidos_ratings.items():
            if lib_leido in conexiones:
                score_final += rating * conexiones[lib_leido]
                
        scores[candidato] = score_final
        
    return scores


def ranker_popularidad(id_lector, candidatos, cache):
    """
    Asigna a cada candidato su score de popularidad global.
    Si el libro es nuevo/desconocido, le asigna la media global.
    """
    pop_dict = cache.get("dict_popularidad_score", {})
    media_global = cache.get("media_global", 0.0)
    
    scores = {}
    for lib in candidatos:
        scores[lib] = pop_dict.get(lib, media_global)
        
    return scores


def ranker_filtrado_colaborativo_dot_product(id_lector, candidatos, cache):
    """
    Filtrado Colaborativo 1: Multiplicación completa de matrices (Producto Punto / Similitud).
    Utiliza todas las features de los vectores.
    """
    m_lectores = cache.get("matriz_lectores")
    m_libros = cache.get("matriz_libros")
    
    if m_lectores is None or m_libros is None or len(candidatos) == 0:
        return {lib: 0.0 for lib in candidatos}
        
    # Cold-Start
    if id_lector not in m_lectores.index:
        return {lib: 0.0 for lib in candidatos}
        
    vector_lector = m_lectores.loc[id_lector]
    
    # Filtramos los candidatos que realmente existen en la matriz de libros
    candidatos_validos = list(set(candidatos).intersection(m_libros.index))
    
    if not candidatos_validos:
        return {lib: 0.0 for lib in candidatos}
        
    matriz_candidatos = m_libros.loc[candidatos_validos]
    
    # Multiplicación directa de la matriz por el vector (Calcula el score total)
    scores_series = matriz_candidatos.dot(vector_lector)
    scores_dict = scores_series.to_dict()
    
    # Rellenamos con 0 los candidatos que no tenían vector
    for lib in candidatos:
        if lib not in scores_dict:
            scores_dict[lib] = 0.0
            
    return scores_dict


def ranker_filtrado_colaborativo_promedio_scores(id_lector, candidatos, cache):
    """
    Filtrado Colaborativo 2: Promedio específico de Scores (Género, Autor, Año).
    Aísla las familias de columnas para promediarlas con el mismo peso (1/3).
    """
    m_lectores = cache.get("matriz_lectores")
    m_libros = cache.get("matriz_libros")
    
    if m_lectores is None or m_libros is None or len(candidatos) == 0:
        return {lib: 0.0 for lib in candidatos}
        
    if id_lector not in m_lectores.index:
        return {lib: 0.0 for lib in candidatos}
        
    vector_lector = m_lectores.loc[id_lector]
    
    candidatos_validos = list(set(candidatos).intersection(m_libros.index))
    if not candidatos_validos:
        return {lib: 0.0 for lib in candidatos}
        
    matriz_candidatos = m_libros.loc[candidatos_validos]
    
    # 1. Identificamos las columnas de cada familia
    cols_genero = [c for c in m_libros.columns if c.startswith('generobook_')]
    cols_autor = [c for c in m_libros.columns if c.startswith('autor_')]
    cols_anio = [c for c in m_libros.columns if c.startswith('anio_edicion_')]
    
    # 2. Calculamos los scores parciales (Sub-productos punto)
    score_genero = matriz_candidatos[cols_genero].dot(vector_lector[cols_genero])
    score_autor = matriz_candidatos[cols_autor].dot(vector_lector[cols_autor])
    score_anio = matriz_candidatos[cols_anio].dot(vector_lector[cols_anio])
    
    # 3. Promediamos
    score_final = (score_genero + score_autor + score_anio) / 3.0
    scores_dict = score_final.to_dict()
    
    # 4. Rellenamos faltantes
    for lib in candidatos:
        if lib not in scores_dict:
            scores_dict[lib] = 0.0
            
    return scores_dict




def _generar_scores_lgbm(id_lector, candidatos, cache, key_modelo, key_features):
    """
    Función interna (helper) para ejecutar cualquier modelo LightGBM.
    No se llama directamente desde el evaluador, es usada por los rankers específicos.
    """
    modelo = cache.get(key_modelo)
    features_modelo = cache.get(key_features)
    df_train = cache.get("df_train_features")
    
    # Manejo de seguridad si falta algún modelo o dato
    if modelo is None or df_train is None or len(candidatos) == 0:
        return {lib: 0.0 for lib in candidatos}
        
    # 1. Armamos el DataFrame crudo de candidatos
    df_candidatos = pd.DataFrame({
        "id_lector": [id_lector] * len(candidatos),
        "id_libro": candidatos
    })
    
    # 2. Aplicamos Feature Engineering (inyectando la historia)
    df_candidatos_feat = pipeline_feature_engineering(df_candidatos, state="test", df_historia=df_train)
    
    # 3. Filtramos las features que usa el modelo
    # (Por seguridad verificamos que la feature se haya generado)
    features_validas = [f for f in features_modelo if f in df_candidatos_feat.columns]
    X_candidatos = df_candidatos_feat[features_validas].copy()
    
    # 4. Alineamos las variables categóricas (Evita que LightGBM explote por categorías nuevas)
    cat_features_eval = [c for c in CAT_FEATURES if c in features_validas]
    for c in cat_features_eval:
        if c in df_train.columns:
            # Forzamos a que tenga exactamente las mismas categorías que el dataset de entrenamiento
            X_candidatos[c] = pd.Categorical(X_candidatos[c], categories=df_train[c].cat.categories)
            
    # 5. Predecimos y empaquetamos en el diccionario estándar
    y_pred = modelo.predict(X_candidatos)
    return dict(zip(candidatos, y_pred))


def ranker_lgbm_regressor(id_lector, candidatos, cache):
    """
    Ranker basado en el modelo LightGBM Regressor.
    """
    return _generar_scores_lgbm(id_lector, candidatos, cache, "modelo_lgbm_reg", "features_lgbm_reg")


def ranker_lgbm_ranker(id_lector, candidatos, cache):
    """
    Ranker basado en el modelo LightGBM Ranker (LambdaMART).
    """
    return _generar_scores_lgbm(id_lector, candidatos, cache, "modelo_lgbm_ranker", "features_lgbm_ranker")

def ranker_ease(id_lector, candidatos, cache):
    """Ranker End-to-End usando los scores matemáticos de EASE."""
    modelo_ease = cache.get('modelo_ease')
    
    if modelo_ease is None:
        return {c: -99999 for c in candidatos}
        
    # Le pedimos a EASE un top holgado para asegurarnos de que haya calculado 
    # los scores de los candidatos que trajo el retrieval
    dict_scores = modelo_ease.predecir_usuario_con_scores(id_lector, top_n=2000)
    
    # Mapeamos los scores a los candidatos. Si EASE no lo recomendó en su top 2000, le damos un score muy bajo
    return {candidato: dict_scores.get(candidato, -99999) for candidato in candidatos}