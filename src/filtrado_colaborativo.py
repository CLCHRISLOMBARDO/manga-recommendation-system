import pandas as pd
import sqlite3
import re
import unicodedata
import os
import contextlib

from src.config import BASE_DB, paises_es, countries_en, OUTPUTS_FILTRADO_COLAB

# --- CONSTANTES SQL PARA REUTILIZAR Y MANTENER EL CÓDIGO LIMPIO ---
SQL_CASE_ANIO = """
    CASE
        WHEN anio_edicion < 1960 THEN 'viejos'
        WHEN anio_edicion >= 1960 AND anio_edicion <= 1969 THEN '1960s'
        WHEN anio_edicion >= 1970 AND anio_edicion <= 1979 THEN '1970s'
        WHEN anio_edicion >= 1980 AND anio_edicion <= 1989 THEN '1980s'
        WHEN anio_edicion >= 1990 AND anio_edicion <= 1999 THEN '1990s'
        WHEN anio_edicion >= 2000 AND anio_edicion <= 2009 THEN '2000s'
        WHEN anio_edicion >= 2010 AND anio_edicion <= 2019 THEN '2010s'
        WHEN anio_edicion >= 2020 AND anio_edicion <= 2029 THEN '2020s'
        ELSE 'futuros'
    END
"""

SQL_CASE_NACIMIENTO = """
    CASE
        WHEN CAST(nacimiento AS NUMERIC) >= 1910 AND CAST(nacimiento AS NUMERIC) < 1960 THEN 'viejos'
        WHEN CAST(nacimiento AS NUMERIC) >= 1960 AND CAST(nacimiento AS NUMERIC) <= 1969 THEN '1960s'
        WHEN CAST(nacimiento AS NUMERIC) >= 1970 AND CAST(nacimiento AS NUMERIC) <= 1979 THEN '1970s'
        WHEN CAST(nacimiento AS NUMERIC) >= 1980 AND CAST(nacimiento AS NUMERIC) <= 1989 THEN '1980s'
        WHEN CAST(nacimiento AS NUMERIC) >= 1990 AND CAST(nacimiento AS NUMERIC) <= 1999 THEN '1990s'
        WHEN CAST(nacimiento AS NUMERIC) >= 2000 AND CAST(nacimiento AS NUMERIC) <= 2009 THEN '2000s'
        WHEN CAST(nacimiento AS NUMERIC) >= 2010 AND CAST(nacimiento AS NUMERIC) <= 2019 THEN '2010s'
        WHEN CAST(nacimiento AS NUMERIC) >= 2020 AND CAST(nacimiento AS NUMERIC) <= 2029 THEN '2020s'
        ELSE 'OTROS'
    END
"""

def _seleccion_tabla(state='entrenamiento'):
    if state == 'entrenamiento':
        return 'interacciones_train'
    return 'interacciones'

def _limpiar_texto(texto):
    if pd.isna(texto): return ""
    texto = str(texto).lower()
    return ''.join(c for c in unicodedata.normalize('NFD', texto) if unicodedata.category(c) != 'Mn')

def _construccion_col_pais(df_todos_pais):
    todos_los_paises = paises_es + countries_en
    patron = '|'.join([re.escape(_limpiar_texto(p)) for p in todos_los_paises])
    columna_limpia = df_todos_pais["vive_en"].apply(_limpiar_texto)

    # Extraemos el país y mapeamos
    mapa_originales = {_limpiar_texto(p): p for p in todos_los_paises}
    df_todos_pais["pais"] = columna_limpia.str.extract(f'({patron})', expand=False).map(mapa_originales)
    
    df_todos_pais_uniques = pd.DataFrame(df_todos_pais['pais'].dropna().unique(), columns=['pais'])
    df_todos_pais_uniques = df_todos_pais_uniques.sort_values(by='pais').reset_index(drop=True)
    return df_todos_pais, df_todos_pais_uniques

def _creacion_dfs_todos(state='entrenamiento'):
    TABLA_INTERACCIONES = _seleccion_tabla(state)
    print(TABLA_INTERACCIONES)
    
    query_todos_generosbook = f"SELECT DISTINCT genero as generobook FROM {TABLA_INTERACCIONES} i JOIN libros lib ON i.id_libro = lib.id_libro ORDER BY 1"
    
    query_todos_autor = f'''
        WITH libros_autor_top100 AS (
            SELECT lib.autor, count(*)
            FROM {TABLA_INTERACCIONES} i JOIN libros lib ON i.id_libro = lib.id_libro
            GROUP BY 1 ORDER BY 2 DESC LIMIT 100
        ),
        libros_categoria_otros AS (
            SELECT *, CASE WHEN autor IN (SELECT autor FROM libros_autor_top100) THEN autor ELSE 'otro' END AS autor_categoria
            FROM libros 
        )
        SELECT DISTINCT autor_categoria
        FROM {TABLA_INTERACCIONES} i JOIN libros_categoria_otros lib ON i.id_libro = lib.id_libro
        ORDER BY 1
    '''
    
    query_todos_anio_edicion = f'''
        WITH libros_anio_edicion AS (
            SELECT {SQL_CASE_ANIO} AS anio_edicion
            FROM {TABLA_INTERACCIONES} i JOIN libros lib ON i.id_libro = lib.id_libro
        )
        SELECT DISTINCT anio_edicion FROM libros_anio_edicion ORDER BY 1
    '''
    
    query_todos_pais = f"SELECT DISTINCT lec.vive_en FROM {TABLA_INTERACCIONES} i JOIN lectores lec ON i.id_lector = lec.id_lector ORDER BY 1"
    
    query_todos_generolector = f"SELECT DISTINCT lec.genero as generolector FROM {TABLA_INTERACCIONES} i JOIN lectores lec ON i.id_lector = lec.id_lector ORDER BY 1"
    
    query_todos_nacimiento = f'''
        WITH nacimiento_table AS (
            SELECT {SQL_CASE_NACIMIENTO} AS nacimiento
            FROM {TABLA_INTERACCIONES} i JOIN lectores lec ON i.id_lector = lec.id_lector
        )
        SELECT DISTINCT nacimiento FROM nacimiento_table ORDER BY 1
    '''

    with contextlib.closing(sqlite3.connect(BASE_DB)) as conn:
        df_todos_generosbook = pd.read_sql_query(query_todos_generosbook, conn).sort_values(by='generobook')
        df_todos_autor = pd.read_sql_query(query_todos_autor, conn).sort_values(by='autor_categoria')
        df_todos_anio_edicion = pd.read_sql_query(query_todos_anio_edicion, conn).sort_values(by='anio_edicion')
        df_todos_pais = pd.read_sql_query(query_todos_pais, conn)
        
        df_todos_pais, df_todos_pais_uniques = _construccion_col_pais(df_todos_pais)
        
        df_todos_generolector = pd.read_sql_query(query_todos_generolector, conn)
        # Corrección de Warning aquí
        df_todos_generolector.loc[df_todos_generolector['generolector'] == '-', 'generolector'] = 'Sexo_Indefinido'
        
        df_todos_nacimiento = pd.read_sql_query(query_todos_nacimiento, conn)
    
    return df_todos_generosbook, df_todos_autor, df_todos_anio_edicion, df_todos_pais, df_todos_pais_uniques, df_todos_generolector, df_todos_nacimiento

def matriz_lectores(state='entrenamiento'):
    file_name = f'matriz_comportamiento_perfiles_lectores_{state}.parquet'
    if os.path.exists(OUTPUTS_FILTRADO_COLAB + file_name):
        print(f"Ya existe matriz de lectores para el state: {state}")
        return pd.read_parquet(OUTPUTS_FILTRADO_COLAB + file_name)
    
    TABLA_INTERACCIONES = _seleccion_tabla(state)
    print(TABLA_INTERACCIONES)
    # 1. Obtenemos el esquema de columnas completo para garantizar consistencia
    df_t_gb, df_t_aut, df_t_anio, df_t_pais, df_t_pais_u, df_t_glect, df_t_nac = _creacion_dfs_todos(state)
    
    cols_gb = [f'generobook_{c}' for c in df_t_gb['generobook']]
    cols_aut = [f'autor_{c}' for c in df_t_aut['autor_categoria']]
    cols_anio = [f'anio_edicion_{c}' for c in df_t_anio['anio_edicion']]
    cols_pais = [f'pais_{c}' for c in df_t_pais_u['pais']]
    cols_glect = [f'generolector_{c}' for c in df_t_glect['generolector']]
    cols_nac = [f'nacimiento_{c}' for c in df_t_nac['nacimiento']]

    with contextlib.closing(sqlite3.connect(BASE_DB)) as conn:
        print("Obteniendo IDs...")
        df_ids = pd.read_sql_query(f'SELECT DISTINCT id_lector FROM {TABLA_INTERACCIONES}', conn)
        
        print("Consultando y pivotando atributos de libros (interacciones)...")
        # Generobook
        q_gb = f'''SELECT i.id_lector, lib.genero as feature, AVG(i.rating)/10.0 as rating
                   FROM {TABLA_INTERACCIONES} i JOIN libros lib ON i.id_libro = lib.id_libro GROUP BY 1, 2'''
        p_gb = pd.read_sql_query(q_gb, conn).pivot_table(index='id_lector', columns='feature', values='rating', fill_value=0)
        p_gb.columns = [f'generobook_{c}' for c in p_gb.columns]
        p_gb = p_gb.reindex(columns=cols_gb, fill_value=0)

        # Autor
        q_aut = f'''
            WITH top_autores AS (
                SELECT lib.autor FROM {TABLA_INTERACCIONES} i JOIN libros lib ON i.id_libro = lib.id_libro
                GROUP BY 1 ORDER BY COUNT(*) DESC LIMIT 100
            )
            SELECT i.id_lector, 
                   CASE WHEN lib.autor IN (SELECT autor FROM top_autores) THEN lib.autor ELSE 'otro' END as feature, 
                   AVG(i.rating)/10.0 as rating
            FROM {TABLA_INTERACCIONES} i JOIN libros lib ON i.id_libro = lib.id_libro GROUP BY 1, 2
        '''
        p_aut = pd.read_sql_query(q_aut, conn).pivot_table(index='id_lector', columns='feature', values='rating', fill_value=0)
        p_aut.columns = [f'autor_{c}' for c in p_aut.columns]
        p_aut = p_aut.reindex(columns=cols_aut, fill_value=0)

        # Año de Edicion
        q_anio = f'''SELECT i.id_lector, {SQL_CASE_ANIO} as feature, AVG(i.rating)/10.0 as rating
                     FROM {TABLA_INTERACCIONES} i JOIN libros lib ON i.id_libro = lib.id_libro GROUP BY 1, 2'''
        p_anio = pd.read_sql_query(q_anio, conn).pivot_table(index='id_lector', columns='feature', values='rating', fill_value=0)
        p_anio.columns = [f'anio_edicion_{c}' for c in p_anio.columns]
        p_anio = p_anio.reindex(columns=cols_anio, fill_value=0)

        print("Consultando y pivotando atributos propios del lector...")
        q_lector = f'''
            SELECT id_lector, vive_en,
                   CASE WHEN genero='-' THEN 'Sexo_Indefinido' ELSE genero END as generolector,
                   {SQL_CASE_NACIMIENTO} as nacimiento
            FROM lectores WHERE id_lector IN (SELECT DISTINCT id_lector FROM {TABLA_INTERACCIONES})
        '''
        df_lectores_info = pd.read_sql_query(q_lector, conn)
        df_lectores_info, _ = _construccion_col_pais(df_lectores_info)
        df_lectores_info['rating'] = 1.0 # Peso por defecto para sus atributos demográficos

        p_glect = df_lectores_info.pivot_table(index='id_lector', columns='generolector', values='rating', fill_value=0)
        p_glect.columns = [f'generolector_{c}' for c in p_glect.columns]
        p_glect = p_glect.reindex(columns=cols_glect, fill_value=0)

        p_pais = df_lectores_info.pivot_table(index='id_lector', columns='pais', values='rating', fill_value=0)
        p_pais.columns = [f'pais_{c}' for c in p_pais.columns]
        p_pais = p_pais.reindex(columns=cols_pais, fill_value=0)

        p_nac = df_lectores_info.pivot_table(index='id_lector', columns='nacimiento', values='rating', fill_value=0)
        p_nac.columns = [f'nacimiento_{c}' for c in p_nac.columns]
        p_nac = p_nac.reindex(columns=cols_nac, fill_value=0)

        print("Ensamblando matriz final de lectores...")
        # Al unir por el index (id_lector), evitamos hacer miles de iteraciones.
        matriz_perfil_resultante = df_ids.set_index('id_lector').join(
            [p_glect, p_pais, p_nac, p_gb, p_aut, p_anio], how='left'
        ).fillna(0).reset_index()

    matriz_perfil_resultante = matriz_perfil_resultante.set_index('id_lector')
    matriz_perfil_resultante.to_parquet(OUTPUTS_FILTRADO_COLAB + file_name)
    return matriz_perfil_resultante
def matriz_libros(state='entrenamiento'):
    file_name = f'matriz_comportamiento_perfiles_libros_{state}.parquet'
    if os.path.exists(OUTPUTS_FILTRADO_COLAB + file_name):
        print(f"Ya existe matriz de libros para el state: {state}")
        return pd.read_parquet(OUTPUTS_FILTRADO_COLAB + file_name)
    
    TABLA_INTERACCIONES = _seleccion_tabla(state)
    print(TABLA_INTERACCIONES)
    # 1. Obtenemos esquema para garantizar columnas idénticas a la versión anterior
    df_t_gb, df_t_aut, df_t_anio, df_t_pais, df_t_pais_u, df_t_glect, df_t_nac = _creacion_dfs_todos(state)
    
    cols_gb = [f'generobook_{c}' for c in df_t_gb['generobook']]
    cols_aut = [f'autor_{c}' for c in df_t_aut['autor_categoria']]
    cols_anio = [f'anio_edicion_{c}' for c in df_t_anio['anio_edicion']]
    cols_pais = [f'pais_{c}' for c in df_t_pais_u['pais']]
    cols_glect = [f'generolector_{c}' for c in df_t_glect['generolector']]
    cols_nac = [f'nacimiento_{c}' for c in df_t_nac['nacimiento']]

    # NOTA: En tu original decía "dataset/data.db". Si BASE_DB es diferente, ajusta esto.
    with contextlib.closing(sqlite3.connect(BASE_DB)) as conn: 
        print("Obteniendo IDs de libros...")
        q_ids = f'SELECT id_libro FROM {TABLA_INTERACCIONES} GROUP BY 1 HAVING COUNT(*) > 10'
        df_ids = pd.read_sql_query(q_ids, conn)

        print("Consultando y pivotando atributos del libro (estáticos)...")
        q_libros_info = f'''
            WITH top_autores AS (
                SELECT lib.autor FROM {TABLA_INTERACCIONES} i JOIN libros lib ON i.id_libro = lib.id_libro
                GROUP BY 1 ORDER BY COUNT(*) DESC LIMIT 100
            )
            SELECT id_libro, genero as generobook,
                   CASE WHEN autor IN (SELECT autor FROM top_autores) THEN autor ELSE 'otro' END as autor_categoria,
                   {SQL_CASE_ANIO} as anio_edicion
            FROM libros WHERE id_libro IN ({q_ids})
        '''
        df_libros_info = pd.read_sql_query(q_libros_info, conn)
        df_libros_info['rating'] = 1.0 # Peso estático de sus propias características

        p_gb = df_libros_info.pivot_table(index='id_libro', columns='generobook', values='rating', fill_value=0)
        p_gb.columns = [f'generobook_{c}' for c in p_gb.columns]
        p_gb = p_gb.reindex(columns=cols_gb, fill_value=0)

        p_aut = df_libros_info.pivot_table(index='id_libro', columns='autor_categoria', values='rating', fill_value=0)
        p_aut.columns = [f'autor_{c}' for c in p_aut.columns]
        p_aut = p_aut.reindex(columns=cols_aut, fill_value=0)

        p_anio = df_libros_info.pivot_table(index='id_libro', columns='anio_edicion', values='rating', fill_value=0)
        p_anio.columns = [f'anio_edicion_{c}' for c in p_anio.columns]
        p_anio = p_anio.reindex(columns=cols_anio, fill_value=0)

        print("Consultando y pivotando perfil de sus lectores (interacciones)...")
        
        # CORRECCIÓN: Separamos generolector y nacimiento para evitar el promedio de promedios
        q_glect = f'''
            SELECT i.id_libro, 
                   CASE WHEN lec.genero='-' THEN 'Sexo_Indefinido' ELSE lec.genero END as feature,
                   AVG(i.rating) as rating
            FROM {TABLA_INTERACCIONES} i JOIN lectores lec ON i.id_lector = lec.id_lector
            WHERE i.id_libro IN ({q_ids})
            GROUP BY 1, 2
        '''
        p_glect = pd.read_sql_query(q_glect, conn).pivot_table(index='id_libro', columns='feature', values='rating', fill_value=0)
        p_glect.columns = [f'generolector_{c}' for c in p_glect.columns]
        p_glect = p_glect.reindex(columns=cols_glect, fill_value=0)

        q_nac = f'''
            SELECT i.id_libro, 
                   {SQL_CASE_NACIMIENTO} as feature,
                   AVG(i.rating) as rating
            FROM {TABLA_INTERACCIONES} i JOIN lectores lec ON i.id_lector = lec.id_lector
            WHERE i.id_libro IN ({q_ids})
            GROUP BY 1, 2
        '''
        p_nac = pd.read_sql_query(q_nac, conn).pivot_table(index='id_libro', columns='feature', values='rating', fill_value=0)
        p_nac.columns = [f'nacimiento_{c}' for c in p_nac.columns]
        p_nac = p_nac.reindex(columns=cols_nac, fill_value=0)

        # País (Este se mantiene igual porque calculaba directo desde la base plana)
        q_pais = f'''
            SELECT i.id_libro, lec.vive_en, i.rating
            FROM {TABLA_INTERACCIONES} i JOIN lectores lec ON i.id_lector = lec.id_lector
            WHERE i.id_libro IN ({q_ids})
        '''
        df_pais = pd.read_sql_query(q_pais, conn)
        df_pais, _ = _construccion_col_pais(df_pais) # Genera la columna 'pais' usando la misma lógica
        df_pais_grp = df_pais.groupby(['id_libro', 'pais'])['rating'].mean().reset_index()
        
        p_pais = df_pais_grp.pivot_table(index='id_libro', columns='pais', values='rating', fill_value=0)
        p_pais.columns = [f'pais_{c}' for c in p_pais.columns]
        p_pais = p_pais.reindex(columns=cols_pais, fill_value=0)

        print("Ensamblando matriz final de libros...")
        matriz_perfil_resultante = df_ids.set_index('id_libro').join(
            [p_glect, p_pais, p_nac, p_gb, p_aut, p_anio], how='left'
        ).fillna(0).reset_index()
        
    matriz_perfil_resultante = matriz_perfil_resultante.set_index('id_libro')    
    matriz_perfil_resultante.to_parquet(OUTPUTS_FILTRADO_COLAB + file_name)
    return matriz_perfil_resultante

def filtrado_colaborativo(state='entrenamiento'):
    # entrenamiento o entrenamiento_final
    m_lectores = matriz_lectores(state)
    m_libros = matriz_libros(state)
    return m_lectores, m_libros

