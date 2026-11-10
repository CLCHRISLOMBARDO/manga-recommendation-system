import pandas as pd
import sqlite3
import re
import unicodedata
import numpy as np
import os
from src.config import OUTPUTS_DATASETS ,BASE_DB , paises_es , countries_en , OUTPUTS_FILTRADO_COLAB






def _seleccion_tabla(state='train'):
    if state =='train':
        TABLA_INTERACCIONES = 'interacciones_train'
    else:
        TABLA_INTERACCIONES = 'interacciones'
    return TABLA_INTERACCIONES

def _limpiar_texto(texto):
    if pd.isna(texto): return ""
    texto = str(texto).lower()
    return ''.join(c for c in unicodedata.normalize('NFD', texto) if unicodedata.category(c) != 'Mn')

def _construccion_col_pais(df_todos_pais):

    todos_los_paises = paises_es + countries_en

    patron = '|'.join([re.escape(_limpiar_texto(p)) for p in todos_los_paises])

    columna_limpia = df_todos_pais["vive_en"].apply(_limpiar_texto)

    # 4. Extraemos el país. (Usamos un mapeo final para devolverle sus tildes/mayúsculas originales)
    mapa_originales = {_limpiar_texto(p): p for p in todos_los_paises}

    df_todos_pais["pais"] = columna_limpia.str.extract(f'({patron})', expand=False).map(mapa_originales)
    df_todos_pais_uniques = pd.DataFrame(df_todos_pais.pais.unique())
    df_todos_pais_uniques.columns = ['pais']
    df_todos_pais_uniques = df_todos_pais_uniques.sort_values(by='pais')
    return df_todos_pais , df_todos_pais_uniques

def _creacion_dfs_todos(state='train'):
    TABLA_INTERACCIONES = _seleccion_tabla(state)
    
    
    # Querys TODOS 
    query_todos_generosbook = f''' select distinct genero as generobook
                                from {TABLA_INTERACCIONES}  i join libros lib on i.id_libro = lib.id_libro
                                order by 1
                                '''
    query_todos_autor = f''' with libros_autor_top100 as (
                                    select lib.autor , count(*)
                                    from {TABLA_INTERACCIONES} i join libros lib on i.id_libro = lib.id_libro
                                    group by 1
                                    order by 2 desc
                                    LIMIT 100
                            ),
                            libros_categoria_otros as (
                                    select *,
                                    case 
                                    when autor in (select autor from libros_autor_top100) then autor else 'otro' end as autor_categoria      
                                    from libros 
                            ) 
                                select distinct autor_categoria
                                from {TABLA_INTERACCIONES} i join libros_categoria_otros lib on i.id_libro = lib.id_libro
                                order by 1
                                '''
    query_todos_anio_edicion = f'''
                            with libros_anio_edicion as (
                            SELECT 
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
                            END AS anio_edicion
                    from {TABLA_INTERACCIONES} i join libros lib on i.id_libro = lib.id_libro)
                    select distinct anio_edicion from libros_anio_edicion order by 1
                            '''
    query_todos_pais = f'''select  distinct lec.vive_en  from {TABLA_INTERACCIONES} i join lectores lec on i.id_lector = lec.id_lector order by 1'''
    query_todos_generolector=f'''select distinct lec.genero  as generolector from {TABLA_INTERACCIONES} i join lectores lec on i.id_lector = lec.id_lector order by 1'''
    query_todos_nacimiento = f'''with nacimiento_table as (
                    SELECT
                            CASE
                            WHEN CAST(nacimiento AS NUMERIC) >= 1910 AND  CAST(nacimiento AS NUMERIC) < 1960 THEN 'viejos'
                            WHEN CAST(nacimiento AS NUMERIC) >= 1960 AND CAST(nacimiento AS NUMERIC) <= 1969 THEN '1960s'
                            WHEN CAST(nacimiento AS NUMERIC) >= 1970 AND CAST(nacimiento AS NUMERIC) <= 1979 THEN '1970s'
                            WHEN CAST(nacimiento AS NUMERIC) >= 1980 AND CAST(nacimiento AS NUMERIC) <= 1989 THEN '1980s'
                            WHEN CAST(nacimiento AS NUMERIC) >= 1990 AND CAST(nacimiento AS NUMERIC) <= 1999 THEN '1990s'
                            WHEN CAST(nacimiento AS NUMERIC) >= 2000 AND CAST(nacimiento AS NUMERIC) <= 2009 THEN '2000s'
                            WHEN CAST(nacimiento AS NUMERIC) >= 2010 AND CAST(nacimiento AS NUMERIC) <= 2019 THEN '2010s'
                            WHEN CAST(nacimiento AS NUMERIC) >= 2020 AND CAST(nacimiento AS NUMERIC) <= 2029 THEN '2020s'
                            ELSE 'OTROS'
                            END AS nacimiento
                    from {TABLA_INTERACCIONES} i join lectores lec on i.id_lector = lec.id_lector)
                    select distinct nacimiento from nacimiento_table
                    order by 1'''

    conn = sqlite3.connect(BASE_DB)
    df_todos_generosbook = pd.read_sql_query(query_todos_generosbook , conn).sort_values(by='generobook')
    df_todos_autor = pd.read_sql_query(query_todos_autor , conn).sort_values(by='autor_categoria')
    df_todos_anio_edicion = pd.read_sql_query(query_todos_anio_edicion , conn).sort_values(by='anio_edicion')
    df_todos_pais = pd.read_sql_query(query_todos_pais , conn)
    
    df_todos_pais , df_todos_pais_uniques = _construccion_col_pais(df_todos_pais)
    
    df_todos_generolector = pd.read_sql_query(query_todos_generolector , conn)
    df_todos_generolector.loc[df_todos_generolector['generolector']=='-'] = 'Sexo_Indefinido'
    df_todos_nacimiento = pd.read_sql_query(query_todos_nacimiento , conn)
    conn.close()
    
    return df_todos_generosbook , df_todos_autor,df_todos_anio_edicion , df_todos_pais, df_todos_pais_uniques , df_todos_generolector , df_todos_nacimiento

def matriz_lectores(state='train'):
    '''Llama a las funciones _seleccion_tabla y _creacion_dfs_todos'''
    
    file_name = f'matriz_comportamiento_perfiles_lectores_{state}.parquet'
    if os.path.exists(OUTPUTS_FILTRADO_COLAB + file_name):
        print(f"Ya existe matriz de lectores para el state : {state}")
        matriz_perfil_resutalte=pd.read_parquet(OUTPUTS_FILTRADO_COLAB+file_name)
        return matriz_perfil_resutalte
    
    TABLA_INTERACCIONES = _seleccion_tabla(state)
    # 1. Obtener IDs
    conn = sqlite3.connect(BASE_DB)
    query = f'select distinct id_lector from {TABLA_INTERACCIONES}'
    ids_lectores_train = list(pd.read_sql_query(query, conn).id_lector.values)
    conn.close()

    print(f"Total a procesar: {len(ids_lectores_train)}")

    query_perfil_generobook = f'''SELECT lib.genero as generobook , AVG(i.rating)/10 as rating_generobook
            FROM {TABLA_INTERACCIONES} i join libros lib on i.id_libro = lib.id_libro 
            WHERE id_lector = ? GROUP BY 1 ORDER BY 2 desc'''

    query_perfil_autor = f'''with libros_autor_top100 as (
                select lib.autor , count(*) from {TABLA_INTERACCIONES} i join libros lib on i.id_libro = lib.id_libro
                group by 1 order by 2 desc LIMIT 100
            ),
            libros_categoria_otros as (
                select *, case when autor in (select autor from libros_autor_top100) then autor else 'otro' end as autor_categoria       
                from libros 
            )
            select autor_categoria , AVG(i.rating)/10 as rating_autor 
            from {TABLA_INTERACCIONES} i join libros_categoria_otros lib on i.id_libro = lib.id_libro 
            where id_lector = ? group by 1 order by 2 desc'''
                                
    query_anio_edicion = f'''SELECT CASE
                WHEN anio_edicion < 1960 THEN 'viejos'
                WHEN anio_edicion >= 1960 AND anio_edicion <= 1969 THEN '1960s'
                WHEN anio_edicion >= 1970 AND anio_edicion <= 1979 THEN '1970s'
                WHEN anio_edicion >= 1980 AND anio_edicion <= 1989 THEN '1980s'
                WHEN anio_edicion >= 1990 AND anio_edicion <= 1999 THEN '1990s'
                WHEN anio_edicion >= 2000 AND anio_edicion <= 2009 THEN '2000s'
                WHEN anio_edicion >= 2010 AND anio_edicion <= 2019 THEN '2010s'
                WHEN anio_edicion >= 2020 AND anio_edicion <= 2029 THEN '2020s'
                ELSE 'futuros' END AS anio_edicion,  AVG(i.rating)/10 as rating_anio_edicion
            from {TABLA_INTERACCIONES} i join libros lib on i.id_libro = lib.id_libro
            where id_lector = ? group by 1 order by 2 desc'''

    query_generolector = f'''select distinct lec.genero as generolector 
            from {TABLA_INTERACCIONES} i join lectores lec on i.id_lector = lec.id_lector
            where lec.id_lector =?'''

    query_pais = f'''select distinct lec.vive_en
            from {TABLA_INTERACCIONES} i join lectores lec on i.id_lector = lec.id_lector
            where lec.id_lector =?'''

    query_nacimiento = f'''with nacimiento_table as (
            SELECT CASE
                WHEN CAST(nacimiento AS NUMERIC) >= 1910 AND  CAST(nacimiento AS NUMERIC) < 1960 THEN 'viejos'
                WHEN CAST(nacimiento AS NUMERIC) >= 1960 AND CAST(nacimiento AS NUMERIC) <= 1969 THEN '1960s'
                WHEN CAST(nacimiento AS NUMERIC) >= 1970 AND CAST(nacimiento AS NUMERIC) <= 1979 THEN '1970s'
                WHEN CAST(nacimiento AS NUMERIC) >= 1980 AND CAST(nacimiento AS NUMERIC) <= 1989 THEN '1980s'
                WHEN CAST(nacimiento AS NUMERIC) >= 1990 AND CAST(nacimiento AS NUMERIC) <= 1999 THEN '1990s'
                WHEN CAST(nacimiento AS NUMERIC) >= 2000 AND CAST(nacimiento AS NUMERIC) <= 2009 THEN '2000s'
                WHEN CAST(nacimiento AS NUMERIC) >= 2010 AND CAST(nacimiento AS NUMERIC) <= 2019 THEN '2010s'
                WHEN CAST(nacimiento AS NUMERIC) >= 2020 AND CAST(nacimiento AS NUMERIC) <= 2029 THEN '2020s'
                ELSE 'OTROS' END AS nacimiento
            from {TABLA_INTERACCIONES} i join lectores lec on i.id_lector = lec.id_lector
            where lec.id_lector = ? )
            select distinct nacimiento from nacimiento_table'''

    #LLAMAMOS A TODOS LOS VECTORES DE TODOS
    df_todos_generosbook , df_todos_autor,df_todos_anio_edicion , df_todos_pais, df_todos_pais_uniques , df_todos_generolector , df_todos_nacimiento = _creacion_dfs_todos(state)
    

    lista_vectores_lectores = []

    conn = sqlite3.connect(BASE_DB)

    for i, id_lector in enumerate(ids_lectores_train):
        print(f"Vamos por el lector {id_lector} : {i}/{len(ids_lectores_train)}")
        
        df_matriz_perfil_i = pd.DataFrame({'id_lector':[id_lector]}).reset_index(drop=True)
        
        df_perfil_generobook = pd.read_sql_query(query_perfil_generobook, conn, params=[id_lector])
        df_perfil_autor = pd.read_sql_query(query_perfil_autor, conn, params=[id_lector])
        df_perfil_anio_eidicion = pd.read_sql_query(query_anio_edicion, conn, params=[id_lector])

        df_perfil_generolector = pd.read_sql_query(query_generolector, conn, params=[id_lector])
        df_perfil_generolector.loc[df_perfil_generolector['generolector']=='-', 'generolector'] = 'Sexo_Indefinido'
        df_perfil_generolector['rating_generolector'] = 1.0

        df_perfil_pais = pd.read_sql_query(query_pais, conn, params=[id_lector])
        df_perfil_pais = df_perfil_pais.merge(df_todos_pais, on='vive_en', how='inner')
        df_perfil_pais_unique = df_perfil_pais[['pais']].copy() # .copy() evita warnings
        df_perfil_pais_unique['rating_pais'] = 1.0

        df_perfil_nacimiento = pd.read_sql_query(query_nacimiento, conn, params=[id_lector])
        df_perfil_nacimiento['rating_nacimiento'] = 1.0
        
        vector_vertical_generolector  = df_todos_generolector.merge(df_perfil_generolector, how='left', on='generolector')
        vector_vertical_generolector.loc[vector_vertical_generolector['rating_generolector'].isna(),'rating_generolector'] = 0.0
        vector_horizontal_generolector = vector_vertical_generolector.set_index('generolector').T.reset_index(drop=True)
        vector_horizontal_generolector.columns = [f'generolector_{c}' for c in vector_horizontal_generolector.columns]

        vector_vertical_pais  = df_todos_pais_uniques.merge(df_perfil_pais_unique, how='left', on='pais')
        vector_vertical_pais.loc[vector_vertical_pais['rating_pais'].isna(),'rating_pais'] = 0.0
        vector_horizontal_pais = vector_vertical_pais.set_index('pais').T.reset_index(drop=True)
        vector_horizontal_pais.columns = [f'pais_{c}' for c in vector_horizontal_pais.columns]

        vector_vertical_nacimiento  = df_todos_nacimiento.merge(df_perfil_nacimiento, how='left', on='nacimiento')
        vector_vertical_nacimiento.loc[vector_vertical_nacimiento['rating_nacimiento'].isna(),'rating_nacimiento'] = 0.0
        vector_horizontal_nacimiento = vector_vertical_nacimiento.set_index('nacimiento').T.reset_index(drop=True)
        vector_horizontal_nacimiento.columns = [f'nacimiento_{c}' for c in vector_horizontal_nacimiento.columns]
        
        vector_vertical_generobook = df_todos_generosbook.merge(df_perfil_generobook, how='left', on='generobook')
        vector_vertical_generobook.loc[vector_vertical_generobook['rating_generobook'].isna(), 'rating_generobook'] = 0.0
        vector_horizontal_generobook = vector_vertical_generobook.set_index('generobook').T.reset_index(drop=True)
        vector_horizontal_generobook.columns = [f'generobook_{c}' for c in vector_horizontal_generobook.columns]

        vector_vertical_autor = df_todos_autor.merge(df_perfil_autor, how='left', on='autor_categoria')
        vector_vertical_autor.loc[vector_vertical_autor['rating_autor'].isna(),'rating_autor'] = 0.0
        vector_horizontal_autor = vector_vertical_autor.set_index('autor_categoria').T.reset_index(drop=True)
        vector_horizontal_autor.columns = [f'autor_{c}' for c in vector_horizontal_autor.columns]

        vector_vertical_anio_edicion = df_todos_anio_edicion.merge(df_perfil_anio_eidicion, how='left', on='anio_edicion')
        vector_vertical_anio_edicion.loc[vector_vertical_anio_edicion['rating_anio_edicion'].isna(),'rating_anio_edicion'] = 0.0
        vector_horizontal_anio_edicion = vector_vertical_anio_edicion.set_index('anio_edicion').T.reset_index(drop=True)
        vector_horizontal_anio_edicion.columns = [f'anio_edicion_{c}' for c in vector_horizontal_anio_edicion.columns]

        vector_lector = pd.concat([
            df_matriz_perfil_i,
            vector_horizontal_generolector,
            vector_horizontal_pais,
            vector_horizontal_nacimiento,
            vector_horizontal_generobook,
            vector_horizontal_autor,
            vector_horizontal_anio_edicion
        ], axis=1)
        
        lista_vectores_lectores.append(vector_lector)

    conn.close()

    matriz_perfil_resutalte = pd.concat(lista_vectores_lectores, axis=0, ignore_index=True)
    
    
    # Guardamos la matriz
    matriz_perfil_resutalte.to_parquet(OUTPUTS_FILTRADO_COLAB+file_name)
    
    return matriz_perfil_resutalte

def matriz_libros(state='train'):
    '''Llama a las funciones _seleccion_tabla y _creacion_dfs_todos'''
    
    file_name = f'matriz_comportamiento_perfiles_libros_{state}.parquet'
    if os.path.exists(OUTPUTS_FILTRADO_COLAB + file_name):
        print(f"Ya existe matriz de lectores para el state : {state}")
        matriz_perfil_resutalte=pd.read_parquet(OUTPUTS_FILTRADO_COLAB+file_name)
        return matriz_perfil_resutalte
    
    TABLA_INTERACCIONES = _seleccion_tabla(state)


    # 1. Obtener IDs
    conn = sqlite3.connect("dataset/data.db")
    query = f'select  id_libro from {TABLA_INTERACCIONES} group by 1 having count(*) > 10'
    ids_libros_train = list(pd.read_sql_query(query, conn).id_libro.values)
    conn.close()

    print(f"Total a procesar: {len(ids_libros_train)}")

    query_perfil_generobook = f'''SELECT distinct genero as generobook , cast(1.0 as numeric) as rating_generobook
            FROM libros
            WHERE id_libro = ? '''

    query_perfil_autor = f'''with libros_autor_top100 as (
                select lib.autor , count(*) from {TABLA_INTERACCIONES} i join libros lib on i.id_libro = lib.id_libro
                group by 1 order by 2 desc LIMIT 100
            ),
            libros_categoria_otros as (
                select *, case when autor in (select autor from libros_autor_top100) then autor else 'otro' end as autor_categoria       
                from libros 
            )
            select distinct autor_categoria , cast(1.0 as numeric) as rating_autor 
            from libros_categoria_otros
            where id_libro = ?'''
                                
    query_anio_edicion = f'''with anio_edicion_table as 
                (SELECT CASE
                WHEN anio_edicion < 1960 THEN 'viejos'
                WHEN anio_edicion >= 1960 AND anio_edicion <= 1969 THEN '1960s'
                WHEN anio_edicion >= 1970 AND anio_edicion <= 1979 THEN '1970s'
                WHEN anio_edicion >= 1980 AND anio_edicion <= 1989 THEN '1980s'
                WHEN anio_edicion >= 1990 AND anio_edicion <= 1999 THEN '1990s'
                WHEN anio_edicion >= 2000 AND anio_edicion <= 2009 THEN '2000s'
                WHEN anio_edicion >= 2010 AND anio_edicion <= 2019 THEN '2010s'
                WHEN anio_edicion >= 2020 AND anio_edicion <= 2029 THEN '2020s'
                ELSE 'futuros' END AS anio_edicion
            from libros 
            where id_libro = ? )
            select distinct anio_edicion, cast(1.0 as numeric) as rating_anio_edicion
            from anio_edicion_table
            '''

    query_generolector = f'''select  lec.genero as generolector, AVG(i.rating) as  rating_generolector
            from {TABLA_INTERACCIONES} i join lectores lec on i.id_lector = lec.id_lector
            where id_libro =?
            group by 1 
            order by 2 desc'''

    query_pais = f'''select i.id_lector, lec.vive_en ,i.rating
            from {TABLA_INTERACCIONES} i join lectores lec on i.id_lector = lec.id_lector
            where id_libro =?
            ''' # Este lo tenemos que transformar con la funcion de python que tenemos

    query_nacimiento = f'''with nacimiento_table as (
            SELECT rating,
            CASE
                WHEN CAST(nacimiento AS NUMERIC) >= 1910 AND  CAST(nacimiento AS NUMERIC) < 1960 THEN 'viejos'
                WHEN CAST(nacimiento AS NUMERIC) >= 1960 AND CAST(nacimiento AS NUMERIC) <= 1969 THEN '1960s'
                WHEN CAST(nacimiento AS NUMERIC) >= 1970 AND CAST(nacimiento AS NUMERIC) <= 1979 THEN '1970s'
                WHEN CAST(nacimiento AS NUMERIC) >= 1980 AND CAST(nacimiento AS NUMERIC) <= 1989 THEN '1980s'
                WHEN CAST(nacimiento AS NUMERIC) >= 1990 AND CAST(nacimiento AS NUMERIC) <= 1999 THEN '1990s'
                WHEN CAST(nacimiento AS NUMERIC) >= 2000 AND CAST(nacimiento AS NUMERIC) <= 2009 THEN '2000s'
                WHEN CAST(nacimiento AS NUMERIC) >= 2010 AND CAST(nacimiento AS NUMERIC) <= 2019 THEN '2010s'
                WHEN CAST(nacimiento AS NUMERIC) >= 2020 AND CAST(nacimiento AS NUMERIC) <= 2029 THEN '2020s'
                ELSE 'OTROS' END AS nacimiento
            from {TABLA_INTERACCIONES} i join lectores lec on i.id_lector = lec.id_lector
            where i.id_libro = ? )
            select  nacimiento,AVG(rating) as rating_nacimiento from nacimiento_table group by 1 order by 2'''
    
    #LLAMAMOS A TODOS LOS VECTORES DE TODOS
    df_todos_generosbook , df_todos_autor,df_todos_anio_edicion , df_todos_pais, df_todos_pais_uniques , df_todos_generolector , df_todos_nacimiento = _creacion_dfs_todos(state)

    lista_vectores_lectores = []

    conn = sqlite3.connect('dataset/data.db')

    for i, id_libro in enumerate(ids_libros_train):
        print(f"Vamos por el libro {id_libro} : {i}/{len(ids_libros_train)}")
        
        df_matriz_perfil_i = pd.DataFrame({'id_libro':[id_libro]}).reset_index(drop=True)
        
        df_perfil_generobook = pd.read_sql_query(query_perfil_generobook, conn, params=[id_libro])
        df_perfil_autor = pd.read_sql_query(query_perfil_autor, conn, params=[id_libro])
        df_perfil_anio_eidicion = pd.read_sql_query(query_anio_edicion, conn, params=[id_libro])

        df_perfil_generolector = pd.read_sql_query(query_generolector, conn, params=[id_libro])
        df_perfil_generolector.loc[df_perfil_generolector['generolector']=='-', 'generolector'] = 'Sexo_Indefinido'

        df_perfil_pais = pd.read_sql_query(query_pais, conn, params=[id_libro])
        df_perfil_pais = df_perfil_pais.merge(df_todos_pais, on='vive_en', how='inner')
        df_perfil_pais_unique = df_perfil_pais.groupby('pais').agg(
            rating_pais=('rating','mean')
        ).reset_index()
        
        df_perfil_nacimiento = pd.read_sql_query(query_nacimiento, conn, params=[id_libro])
        
        vector_vertical_generolector  = df_todos_generolector.merge(df_perfil_generolector, how='left', on='generolector')
        vector_vertical_generolector.loc[vector_vertical_generolector['rating_generolector'].isna(),'rating_generolector'] = 0.0
        vector_horizontal_generolector = vector_vertical_generolector.set_index('generolector').T.reset_index(drop=True)
        vector_horizontal_generolector.columns = [f'generolector_{c}' for c in vector_horizontal_generolector.columns]

        vector_vertical_pais  = df_todos_pais_uniques.merge(df_perfil_pais_unique, how='left', on='pais')
        vector_vertical_pais.loc[vector_vertical_pais['rating_pais'].isna(),'rating_pais'] = 0.0
        vector_horizontal_pais = vector_vertical_pais.set_index('pais').T.reset_index(drop=True)
        vector_horizontal_pais.columns = [f'pais_{c}' for c in vector_horizontal_pais.columns]

        vector_vertical_nacimiento  = df_todos_nacimiento.merge(df_perfil_nacimiento, how='left', on='nacimiento')
        vector_vertical_nacimiento.loc[vector_vertical_nacimiento['rating_nacimiento'].isna(),'rating_nacimiento'] = 0.0
        vector_horizontal_nacimiento = vector_vertical_nacimiento.set_index('nacimiento').T.reset_index(drop=True)
        vector_horizontal_nacimiento.columns = [f'nacimiento_{c}' for c in vector_horizontal_nacimiento.columns]
        
        vector_vertical_generobook = df_todos_generosbook.merge(df_perfil_generobook, how='left', on='generobook')
        vector_vertical_generobook.loc[vector_vertical_generobook['rating_generobook'].isna(), 'rating_generobook'] = 0.0
        vector_horizontal_generobook = vector_vertical_generobook.set_index('generobook').T.reset_index(drop=True)
        vector_horizontal_generobook.columns = [f'generobook_{c}' for c in vector_horizontal_generobook.columns]

        vector_vertical_autor = df_todos_autor.merge(df_perfil_autor, how='left', on='autor_categoria')
        vector_vertical_autor.loc[vector_vertical_autor['rating_autor'].isna(),'rating_autor'] = 0.0
        vector_horizontal_autor = vector_vertical_autor.set_index('autor_categoria').T.reset_index(drop=True)
        vector_horizontal_autor.columns = [f'autor_{c}' for c in vector_horizontal_autor.columns]

        vector_vertical_anio_edicion = df_todos_anio_edicion.merge(df_perfil_anio_eidicion, how='left', on='anio_edicion')
        vector_vertical_anio_edicion.loc[vector_vertical_anio_edicion['rating_anio_edicion'].isna(),'rating_anio_edicion'] = 0.0
        vector_horizontal_anio_edicion = vector_vertical_anio_edicion.set_index('anio_edicion').T.reset_index(drop=True)
        vector_horizontal_anio_edicion.columns = [f'anio_edicion_{c}' for c in vector_horizontal_anio_edicion.columns]

        vector_lector = pd.concat([
            df_matriz_perfil_i,
            vector_horizontal_generolector,
            vector_horizontal_pais,
            vector_horizontal_nacimiento,
            vector_horizontal_generobook,
            vector_horizontal_autor,
            vector_horizontal_anio_edicion
        ], axis=1)
        
        lista_vectores_lectores.append(vector_lector)

    conn.close()

    matriz_perfil_resutalte = pd.concat(lista_vectores_lectores, axis=0, ignore_index=True)
    
    # Guardamos la matriz
    matriz_perfil_resutalte.to_parquet(OUTPUTS_FILTRADO_COLAB+file_name)
    
    return matriz_perfil_resutalte

def filtrado_colaborativo(state='train'):
    
    m_lectores=matriz_lectores(state)
    m_libros = matriz_libros(state)
    
    return m_lectores , m_libros

    