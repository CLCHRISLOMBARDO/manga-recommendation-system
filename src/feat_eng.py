import pandas as pd
import sqlite3

from src.config import BASE_DB

def _fing_mergeo_features_bases(df: pd.DataFrame, state="entrenamiento"):
    con = sqlite3.connect(BASE_DB)
    
    # 1. Traemos los catálogos por separado
    libros = pd.read_sql_query("""
        SELECT id_libro, titulo, autor, genero as genero_libro, 
             CAST(anio_edicion AS integer) AS anio_edicion,
             CAST(rating_promedio AS REAL) as rating_promedio,
             CAST(cantidad_votos AS integer) AS cantidad_votos,
             CAST(volumenes AS integer) AS volumenes,
             CAST(capitulos AS integer) AS capitulos,
             CAST(ranking AS integer) AS ranking,
             CAST(en_emision AS integer) AS en_emision
        FROM libros
    """, con)
    
    lectores = pd.read_sql_query("""
        SELECT id_lector, genero, cast(vive_en as text) as vive_en, 
               cast(nacimiento as INTEGER) as nacimiento 
        FROM lectores
    """, con)
    con.close()
    
    # 2. Hacemos el cruce de forma independiente (¡Clave para los candidatos!)
    df_mergeo = pd.merge(df, libros, on="id_libro", how="left")
    df_mergeo = pd.merge(df_mergeo, lectores, on="id_lector", how="left")
    
    # 3. Solo si estamos entrenando, vamos a buscar la fecha real de la interacción
    if state not in ["entrenamiento", "entrenamiento_final"]:
        df_mergeo["fecha_interaccion"] = pd.NaT

    return df_mergeo



def _fing_volumen_lectura_por_lector(df: pd.DataFrame, state="entrenamiento", df_historia=None) -> pd.DataFrame:
    col = "volumen_lectura_por_lector"
    if state != "test":
        print(f"-> Aplicando: {col} para el estado : {state}")
    if state in ["entrenamiento","entrenamiento_final"]:
        df[col] = df.groupby("id_lector")["id_libro"].transform("count")

    elif state in ["test", "prediccion_final"]:
        if df_historia is None:
            file_name = "df_train_features.parquet" if state == "test" else "df_total_features.parquet"
            df_historia = pd.read_parquet(f"outputs/datasets/{file_name}")
            
        df = pd.merge(df, df_historia[['id_lector', col]].drop_duplicates(subset=["id_lector"]), on="id_lector", how='left')
    
    df[col] = df[col].fillna(0)
    return df


def _fing_rating_promedio_crudo_por_lector(df: pd.DataFrame, state="entrenamiento", df_historia=None) -> pd.DataFrame:
    col = "rating_promedio_crudo_por_lector"

    if state != "test":
        print(f"-> Aplicando: {col} para el estado : {state}")
    if state in ["entrenamiento","entrenamiento_final"]:
        df[col] = df.groupby("id_lector")["rating"].transform("mean")

    elif state in ["test", "prediccion_final"]:
        if df_historia is None:
            file_name = "df_train_features.parquet" if state == "test" else "df_total_features.parquet"
            df_historia = pd.read_parquet(f"outputs/datasets/{file_name}")
            
        df = pd.merge(df, df_historia[['id_lector', col]].drop_duplicates(subset=["id_lector"]), on="id_lector", how='left')
    return df


def _fing_desviacion_estandar_rating_por_lector(df: pd.DataFrame, state="entrenamiento", df_historia=None) -> pd.DataFrame:
    col = "desviacion_estandar_rating_por_lector"

    if state != "test":
        print(f"-> Aplicando: {col} para el estado : {state}")
    if state in ["entrenamiento","entrenamiento_final"]:
        std = df.groupby("id_lector")["rating"].transform(lambda x: x.std(ddof=0))
        df[col] = std.fillna(0.0).round(4)

    elif state in ["test", "prediccion_final"]:
        if df_historia is None:
            file_name = "df_train_features.parquet" if state == "test" else "df_total_features.parquet"
            df_historia = pd.read_parquet(f"outputs/datasets/{file_name}")
            
        df = pd.merge(df, df_historia[['id_lector', col]].drop_duplicates(subset=["id_lector"]), on="id_lector", how='left')  

    df[col] = df[col].fillna(0)
    return df


def _fing_genero_favorito_por_lector(df: pd.DataFrame, state ="entrenamiento", df_historia=None) -> pd.DataFrame:
    col = "genero_favorito_por_lector"

    if state != "test":
        print(f"-> Aplicando: {col} para el estado : {state}")
    if state in ["entrenamiento","entrenamiento_final"]:
        validos = df[df["genero_libro"].notna() & (df["genero_libro"].astype(str).str.strip() != "")]
        top_genero = (
            validos.groupby("id_lector")["genero_libro"]
            .agg(lambda x: x.mode().iloc[0] if not x.empty else "DESCONOCIDO")
            .reset_index(name=col)
        )
        df = df.merge(top_genero, on="id_lector", how="left")
        df[col] = df[col].fillna("DESCONOCIDO")

    elif state in ["test", "prediccion_final"]:
        if df_historia is None:
            file_name = "df_train_features.parquet" if state == "test" else "df_total_features.parquet"
            df_historia = pd.read_parquet(f"outputs/datasets/{file_name}")
            
        df = pd.merge(df, df_historia[['id_lector', col]].drop_duplicates(subset=["id_lector"]), on="id_lector", how='left') 

    return df


def _fing_antiguedad_lectura_por_lector(df: pd.DataFrame, state = "entrenamiento", df_historia=None) -> pd.DataFrame:
    col="ratio_libros_antiguos_vs_nuevos_por_lector"
    if state != "test":
        print(f"-> Aplicando: {col} para el estado : {state}")

    if state in ["entrenamiento","entrenamiento_final"]:
        es_antiguo = (df["anio_edicion"] < 2010).astype(int)
        es_nuevo = (df["anio_edicion"] >= 2010).astype(int)

        stats_anio = df[["id_lector"]].copy()
        stats_anio["antiguo"] = es_antiguo
        stats_anio["nuevo"] = es_nuevo

        agrupado = stats_anio.groupby("id_lector").agg(
            cant_antiguos=("antiguo", "sum"),
            cant_nuevos=("nuevo", "sum"),
        )
        agrupado[col] = (
            agrupado["cant_antiguos"] / (agrupado["cant_nuevos"] + 1.0)
        ).round(4) 
        df = df.merge(agrupado[[col]], on="id_lector", how="left")

    elif state in ["test", "prediccion_final"]:
        if df_historia is None:
            file_name = "df_train_features.parquet" if state == "test" else "df_total_features.parquet"
            df_historia = pd.read_parquet(f"outputs/datasets/{file_name}")
            
        df = pd.merge(df, df_historia[['id_lector', col]].drop_duplicates(subset=["id_lector"]), on="id_lector", how='left') 
    
    df[col] = df[col].fillna(0)
    return df


def _fing_rating_promedio_te_por_lector(df: pd.DataFrame, m_weight: float = 50.0, state="entrenamiento", df_historia=None) -> pd.DataFrame:
    col = "rating_promedio_te_por_lector"

    if state != "test":
        print(f"-> Aplicando: {col} para el estado : {state}")
    if state in ["entrenamiento","entrenamiento_final"]:
        mu = df["rating"].mean()
        
        stats = df.groupby("id_lector")["rating"].agg(n="count", mean_y="mean").reset_index()
        stats[col] = (
            (stats["n"] * stats["mean_y"] + m_weight * mu) / (stats["n"] + m_weight)
        ).round(4)

        df = df.merge(stats[["id_lector", col]], on="id_lector", how="left")

    elif state in ["test", "prediccion_final"]:
        if df_historia is None:
            file_name = "df_train_features.parquet" if state == "test" else "df_total_features.parquet"
            df_historia = pd.read_parquet(f"outputs/datasets/{file_name}")
            
        mu = df_historia["rating"].mean()
        df = pd.merge(df, df_historia[['id_lector', col]].drop_duplicates(subset=["id_lector"]), on="id_lector", how='left') 
  
    df[col] = df[col].fillna(mu)
    return df

def _fing_pct_autor_favorito_por_lector(df: pd.DataFrame, state="entrenamiento", df_historia=None) -> pd.DataFrame:
    col = "pct_autor_favorito_por_lector"
    if state != "test":
        print(f"-> Aplicando: {col} para el estado : {state}")
    if state in ["entrenamiento","entrenamiento_final"]:
        total_leidos = df.groupby("id_lector").size()
        max_por_autor = df.groupby(["id_lector", "autor"]).size().groupby(level="id_lector").max()
        df_pct = (max_por_autor / total_leidos).rename(col).reset_index()
        df = df.merge(df_pct, on="id_lector", how="left")

    elif state in ["test", "prediccion_final"]:
        if df_historia is None:
            file_name = "df_train_features.parquet" if state == "test" else "df_total_features.parquet"
            df_historia = pd.read_parquet(f"outputs/datasets/{file_name}")
            
        df = pd.merge(df, df_historia[['id_lector', col]].drop_duplicates(subset=["id_lector"]), on="id_lector", how='left')    
    
    df[col] = df[col].fillna(0)
    return df

def _fing_pct_genero_favorito_por_lector(df: pd.DataFrame, state="entrenamiento", df_historia=None) -> pd.DataFrame:
    col = "pct_genero_favorito_por_lector"
    if state != "test":
        print(f"-> Aplicando: {col} para el estado : {state}")
    
    if state in ["entrenamiento", "entrenamiento_final"]:
        total_leidos = df.groupby("id_lector").size()
        max_por_genero = df.groupby(["id_lector", "genero_libro"]).size().groupby(level="id_lector").max()
        df_pct = (max_por_genero / total_leidos).rename(col).reset_index()
        df = df.merge(df_pct, on="id_lector", how="left")

    elif state in ["test", "prediccion_final"]:
        if df_historia is None:
            file_name = "df_train_features.parquet" if state == "test" else "df_total_features.parquet"
            df_historia = pd.read_parquet(f"outputs/datasets/{file_name}")
            
        df = pd.merge(df, df_historia[['id_lector', col]].drop_duplicates(subset=["id_lector"]), on="id_lector", how='left')  
    
    df[col] = df[col].fillna(0.0)
    return df

def _fing_popularidad_por_libro(df:pd.DataFrame, state ="entrenamiento", df_historia=None) -> pd.DataFrame:
    col1 = "popularidad_count_por_libro"
    col2 = "popularidad_rating_mean_por_libro"
    if state != "test":
        print(f"-> Aplicando: {col1} y {col2} para el estado : {state}")

    if state in ["entrenamiento","entrenamiento_final"]:
        states_libro = df.groupby('id_libro').agg(**{
            col1: ('id_lector', 'count'),
            col2: ('rating', 'mean')
        })
        df = pd.merge(df , states_libro , on='id_libro' , how='left')

    elif state in ["test", "prediccion_final"]:
        if df_historia is None:
            file_name = "df_train_features.parquet" if state == "test" else "df_total_features.parquet"
            df_historia = pd.read_parquet(f"outputs/datasets/{file_name}")
            
        df = pd.merge(df, df_historia[['id_libro', col1,col2]].drop_duplicates(subset=["id_libro"]), on="id_libro", how='left')  

    df[col1] = df[col1].fillna(0)
    df[col2] = df[col2].fillna(df[col2].mean())
    return df 


def _fing_rating_promedio_te_por_libro(df: pd.DataFrame, m_weight: float = 50.0, state="entrenamiento", df_historia=None) -> pd.DataFrame:
    col = "rating_promedio_te_por_libro"

    if state != "test":
        print(f"-> Aplicando: {col} para el estado : {state}")
    if state in ["entrenamiento","entrenamiento_final"]:
        mu = df["rating"].mean()
        
        stats = df.groupby("id_libro")["rating"].agg(n="count", mean_y="mean").reset_index()
        stats[col] = (
            (stats["n"] * stats["mean_y"] + m_weight * mu) / (stats["n"] + m_weight)
        ).round(4)

        df = df.merge(stats[["id_libro", col]], on="id_libro", how="left")

    elif state in ["test", "prediccion_final"]:
        if df_historia is None:
            file_name = "df_train_features.parquet" if state == "test" else "df_total_features.parquet"
            df_historia = pd.read_parquet(f"outputs/datasets/{file_name}")
            
        mu = df_historia["rating"].mean()
        df = pd.merge(df, df_historia[['id_libro', col]].drop_duplicates(subset=["id_libro"]), on="id_libro", how='left') 

    df[col] = df[col].fillna(mu)
    return df


def _fing_pct_lecturas_autor_candidato_por_par_libroylector(df: pd.DataFrame, state="entrenamiento", df_historia=None) -> pd.DataFrame:
    col = "pct_lecturas_autor_candidato_por_par_libroylector"
    if state != "test":
        print(f"-> Aplicando: {col} para el estado : {state}")
    
    if state in ["entrenamiento", "entrenamiento_final"]:
        df_historia = df

    elif state in ["test", "prediccion_final"]:
        if df_historia is None:
            file_name = "df_train_features.parquet" if state == "test" else "df_total_features.parquet"
            df_historia = pd.read_parquet(f"outputs/datasets/{file_name}")

    total_leidos = df_historia.groupby("id_lector").size().rename("total_usuario")
    leidos_por_autor = df_historia.groupby(["id_lector", "autor"]).size().rename("leidos_del_autor")
    
    stats_afinidad = pd.merge(leidos_por_autor, total_leidos, left_index=True, right_index=True).reset_index()
    stats_afinidad[col] = (stats_afinidad["leidos_del_autor"] / stats_afinidad["total_usuario"]).round(4)
    
    df = df.merge(stats_afinidad[["id_lector", "autor", col]], on=["id_lector", "autor"], how="left")
    df[col] = df[col].fillna(0.0)
    
    return df

def _fing_pct_lecturas_genero_candidato_por_par_libroylector(df: pd.DataFrame, state="entrenamiento", df_historia=None) -> pd.DataFrame:
    col = "pct_lecturas_genero_candidato_por_par_libroylector"
    if state != "test":
        print(f"-> Aplicando: {col} para el estado : {state}")
    
    if state in ["entrenamiento", "entrenamiento_final"]:
        df_historia = df

    elif state in ["test", "prediccion_final"]:
        if df_historia is None:
            file_name = "df_train_features.parquet" if state == "test" else "df_total_features.parquet"
            df_historia = pd.read_parquet(f"outputs/datasets/{file_name}")
    
    total_leidos = df_historia.groupby("id_lector").size().rename("total_usuario")
    leidos_por_genero = df_historia.groupby(["id_lector", "genero_libro"]).size().rename("leidos_del_genero")
    
    stats_afinidad = pd.merge(leidos_por_genero, total_leidos, left_index=True, right_index=True).reset_index()
    stats_afinidad[col] = (stats_afinidad["leidos_del_genero"] / stats_afinidad["total_usuario"]).round(4)
    
    df = df.merge(stats_afinidad[["id_lector", "genero_libro", col]], on=["id_lector", "genero_libro"], how="left")
    df[col] = df[col].fillna(0.0)
    
    return df

def pipeline_feature_engineering(df: pd.DataFrame, state="entrenamiento", df_historia=None) -> pd.DataFrame:
    """
    Ejecuta en cadena todas las funciones de Feature Engineering.
    """
    if state != "test":
        print("=" * 15 + f" INICIANDO PIPELINE DE FEATURE ENGINEERING del state : {state} " + "=" * 15)
    
    df_out = df.copy()

    # Cadena de transformaciones
    df_out = _fing_mergeo_features_bases(df_out, state)
    #df_out = _fing_pais_lector(df_out, state)
    df_out = _fing_volumen_lectura_por_lector(df_out, state, df_historia=df_historia)
    df_out = _fing_rating_promedio_crudo_por_lector(df_out, state, df_historia=df_historia)
    df_out = _fing_desviacion_estandar_rating_por_lector(df_out, state, df_historia=df_historia)
    df_out = _fing_genero_favorito_por_lector(df_out, state, df_historia=df_historia)
    df_out = _fing_antiguedad_lectura_por_lector(df_out, state, df_historia=df_historia)
    df_out = _fing_rating_promedio_te_por_lector(df_out, state=state, m_weight=50.0, df_historia=df_historia)
    df_out = _fing_pct_autor_favorito_por_lector(df_out, state, df_historia=df_historia)
    df_out = _fing_pct_genero_favorito_por_lector(df_out, state, df_historia=df_historia)
    df_out = _fing_popularidad_por_libro(df_out, state, df_historia=df_historia)
    df_out = _fing_rating_promedio_te_por_libro(df_out, state=state, m_weight=50.0, df_historia=df_historia)
    df_out = _fing_pct_lecturas_autor_candidato_por_par_libroylector(df_out, state, df_historia=df_historia)
    df_out = _fing_pct_lecturas_genero_candidato_por_par_libroylector(df_out, state, df_historia=df_historia)

    # Solo guardamos a disco si estamos entrenando
    if state in ["entrenamiento", "entrenamiento_final"]:
        file_name = "df_train_features.parquet" if state == "entrenamiento" else "df_total_features.parquet"
        print("=" * 15 + " GUARDANDO " + "=" * 15)
        df_out.to_parquet(f"outputs/datasets/{file_name}", index=False)
    
    if state != "test":
        print("=" * 15 + " PIPELINE FINALIZADO CON ÉXITO " + "=" * 15)
    
    return df_out