import sqlite3
import os
import pickle

import src.retrievals as ret
import src.rankers as rnk

DATABASE_FILE = os.path.dirname(__file__) + "/dataset/recomendador_mangas.db"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_FILE = os.path.join(BASE_DIR, "outputs", "cache_final", "cache.pkl")

def sql_select(query, params=None):
    con = sqlite3.connect(DATABASE_FILE)
    con.row_factory = sqlite3.Row # esto es para que devuelva registros en el fetchall
    cur = con.cursor()
    if params:
        res = cur.execute(query, params)
    else:
        res = cur.execute(query)

    ret = res.fetchall()
    con.close()
    return ret

def sql_execute(query, params=None):
    con = sqlite3.connect(DATABASE_FILE)
    cur = con.cursor()
    if params:
        res = cur.execute(query, params)
    else:
        res = cur.execute(query)

    con.commit()
    con.close()
    return res

###

def items_valorados(id_lector):
    query = f"SELECT id_libro FROM interacciones WHERE id_lector = ?"
    rows = sql_select(query, [id_lector])
    return [i["id_libro"] for i in rows]

def items_vistos(id_lector):
    query = f"SELECT id_libro FROM vistos WHERE id_lector = ?"
    rows = sql_select(query, [id_lector])
    return [i["id_libro"] for i in rows]

def items_desconocidos(id_lector):
    query = f"""
        SELECT id_libro
          FROM libros
         WHERE id_libro NOT IN (SELECT id_libro FROM interacciones WHERE id_lector = ?)
           AND id_libro NOT IN (SELECT id_libro FROM vistos WHERE id_lector = ?)
    """
    rows = sql_select(query, [id_lector, id_lector])
    return [i["id_libro"] for i in rows]

def insertar_vistos(id_lector, id_libros):
    for id_libro in id_libros:
        query = f"INSERT INTO vistos (id_lector, id_libro, fecha) VALUES (?, ?, date('now')) ON CONFLICT DO NOTHING;"
        sql_execute(query, [id_lector, id_libro])

def insertar_interaccion(id_lector, id_libro, rating):
    # TODO: hacer UPSERT
    query = f"INSERT INTO interacciones (id_lector, id_libro, rating, fecha) VALUES (?, ?, ?, date('now')) ON CONFLICT DO NOTHING;"
    sql_execute(query, [id_lector, id_libro, rating])

###

def reset_usuario(id_lector):
    sql_execute("DELETE FROM vistos WHERE id_lector = ?", [id_lector])
    sql_execute("DELETE FROM interacciones WHERE id_lector = ?", [id_lector])

###

def recomendador_azar(id_lector):
    rec = sql_select(f"""
        SELECT *
          FROM libros
         WHERE titulo IS NOT NULL
           AND id_libro NOT IN (SELECT id_libro FROM interacciones WHERE id_lector = ?)
           AND id_libro NOT IN (SELECT id_libro FROM vistos WHERE id_lector = ?)
         ORDER BY RANDOM()
    """, [id_lector, id_lector])

    return rec

def recomendador_top_n(id_lector):

    rec = sql_select(f"""
        SELECT libros.*
          FROM libros JOIN interacciones USING (id_libro)
         WHERE rating > 5 -- hiperparámetro
           AND id_libro NOT IN (SELECT id_libro FROM interacciones WHERE id_lector = ?)
           AND id_libro NOT IN (SELECT id_libro FROM vistos WHERE id_lector = ?)
         GROUP BY id_libro
         ORDER BY count(*)*avg(rating) DESC
        """, [id_lector, id_lector])

    return rec


def recomendador_perfil(id_lector):
    rec = sql_select("""
        SELECT *, perfil_genero.cant*0.8 + perfil_autor.cant*0.2 AS score -- 0.8 y 0.2 son hiperparámetros
          FROM libros
          JOIN (SELECT genero, count(*) AS cant
                  FROM libros JOIN interacciones USING (id_libro)
                 WHERE id_lector = ?
                   AND rating > 5
                 GROUP BY 1) AS perfil_genero USING (genero)
          JOIN (SELECT autor, count(*) AS cant
                  FROM libros JOIN interacciones USING (id_libro)
                 WHERE id_lector = ?
                   AND rating > 5 -- hiperparámetro
                 GROUP BY 1) AS perfil_autor USING (autor)
         WHERE titulo IS NOT NULL
           AND id_libro NOT IN (SELECT id_libro FROM interacciones WHERE id_lector = ?)
           AND id_libro NOT IN (SELECT id_libro FROM vistos WHERE id_lector = ?)
         ORDER BY score DESC
    """, [id_lector, id_lector, id_lector, id_lector])
    return rec

def recomendador_cold_starts(id_lector, top_n=9):
    with open(CACHE_FILE, 'rb') as f:
        cache = pickle.load(f)
        
    # Traemos los 100 candidatos
    candidatos = ret.retrieval_hibrido(id_lector, 100, cache)
    scores_predichos_dict = rnk.ranker_popularidad(id_lector, candidatos, cache=cache)
    scores_predichos_sorted = dict(sorted(scores_predichos_dict.items(), key=lambda item: item[1], reverse=True))
    
    # 1. Tomamos TODOS los IDs (los 100 que salieron del ranker)
    todos_ids = list(scores_predichos_sorted.keys())
    if not todos_ids:
            return []
    
    
    # 2. Generar comodines para todos
    placeholders = ','.join(['?'] * len(todos_ids))
    
    # 3. Lógica de ordenamiento para todos
    order_cases = " ".join([f"WHEN {repr(id_libro)} THEN {i}" for i, id_libro in enumerate(todos_ids)])
    
    # 4. Query sin LIMIT
    query = f"""
        SELECT *
        FROM libros
        WHERE id_libro IN ({placeholders})
          AND id_libro NOT IN (SELECT id_libro FROM interacciones WHERE id_lector = ?)
          AND id_libro NOT IN (SELECT id_libro FROM vistos WHERE id_lector = ?)
        ORDER BY CASE id_libro
            {order_cases}
        END
    """
    
    # 5. Ejecutamos la consulta con todos los parámetros
    parametros = todos_ids + [id_lector, id_lector]
    rec = sql_select(query, parametros)
    
    # 6. APLICAMOS EL TOP N AL FINAL
    return rec[:top_n] 

def recomendador_volumen_menor35(id_lector, top_n=9):
    with open('outputs/cache_final/cache.pkl', 'rb') as f:
        cache = pickle.load(f)
        
    # Traemos los 100 candidatos
    candidatos = ret.retrieval_hibrido(id_lector, 100, cache)
    scores_predichos_dict = rnk.ranker_ease(id_lector, candidatos, cache=cache)
    scores_predichos_sorted = dict(sorted(scores_predichos_dict.items(), key=lambda item: item[1], reverse=True))
    
    # 1. Tomamos TODOS los IDs (los 100 que salieron del ranker)
    todos_ids = list(scores_predichos_sorted.keys())
    if not todos_ids:
            return []
    
    
    # 2. Generar comodines para todos
    placeholders = ','.join(['?'] * len(todos_ids))
    
    # 3. Lógica de ordenamiento para todos
    order_cases = " ".join([f"WHEN {repr(id_libro)} THEN {i}" for i, id_libro in enumerate(todos_ids)])
    
    # 4. Query sin LIMIT
    query = f"""
        SELECT *
        FROM libros
        WHERE id_libro IN ({placeholders})
          AND id_libro NOT IN (SELECT id_libro FROM interacciones WHERE id_lector = ?)
          AND id_libro NOT IN (SELECT id_libro FROM vistos WHERE id_lector = ?)
        ORDER BY CASE id_libro
            {order_cases}
        END
    """
    
    # 5. Ejecutamos la consulta con todos los parámetros
    parametros = todos_ids + [id_lector, id_lector]
    rec = sql_select(query, parametros)
    
    # 6. APLICAMOS EL TOP N AL FINAL
    return rec[:top_n]  # Usar rec[:top_n] si rec fuera una lista de diccionarios/tuplas

def recomendador_volumen_mayor35(id_lector, top_n=9):
    with open('outputs/cache_final/cache.pkl', 'rb') as f:
        cache = pickle.load(f)
        
    # Traemos los 100 candidatos
    candidatos = ret.retrieval_q2_mismo_autor(id_lector, 100, cache)
    scores_predichos_dict = rnk.ranker_ease(id_lector, candidatos, cache=cache)
    scores_predichos_sorted = dict(sorted(scores_predichos_dict.items(), key=lambda item: item[1], reverse=True))
    
    # 1. Tomamos TODOS los IDs (los 100 que salieron del ranker)
    todos_ids = list(scores_predichos_sorted.keys())
    if not todos_ids:
            return []
    
    # 2. Generar comodines para todos
    placeholders = ','.join(['?'] * len(todos_ids))
    
    # 3. Lógica de ordenamiento para todos
    order_cases = " ".join([f"WHEN {repr(id_libro)} THEN {i}" for i, id_libro in enumerate(todos_ids)])
    
    # 4. Query sin LIMIT
    query = f"""
        SELECT *
        FROM libros
        WHERE id_libro IN ({placeholders})
          AND id_libro NOT IN (SELECT id_libro FROM interacciones WHERE id_lector = ?)
          AND id_libro NOT IN (SELECT id_libro FROM vistos WHERE id_lector = ?)
        ORDER BY CASE id_libro
            {order_cases}
        END
    """
    
    # 5. Ejecutamos la consulta con todos los parámetros
    parametros = todos_ids + [id_lector, id_lector]
    rec = sql_select(query, parametros)
    
    # 6. APLICAMOS EL TOP N AL FINAL
    return rec[:top_n] # Usar rec[:top_n] si rec fuera una lista de diccionarios/tuplas

    
    

def recomendar(id_lector,N=9):
    """Devuelve N libros recomendados para el lector id_lector"""
    #rec = recomendador_azar(id_lector)

    cant_items_valorados = len(items_valorados(id_lector))
    print(cant_items_valorados)

    if cant_items_valorados <= 3: # hiperparámetro
        rec = recomendador_cold_starts(id_lector)
        algoritmo_usado = "Cold Start (Híbrido + Popularidad)"
        if len(rec) < N:
            rec = rec + recomendador_top_n(id_lector)
        
    elif cant_items_valorados <= 35:
        rec = recomendador_volumen_menor35(id_lector)
        algoritmo_usado = "Volumen Menor 35 (Híbrido + EASE)"

        if len(rec) < N:
            rec = rec + recomendador_top_n(id_lector)
    
    elif cant_items_valorados > 35:
        rec = recomendador_volumen_mayor35(id_lector)
        algoritmo_usado = "Volumen Mayor 35 (Mismo Autor + EASE)"

        if len(rec) < N:
            rec = rec + recomendador_top_n(id_lector)
  
    rec = rec[:N]
    insertar_vistos(id_lector, [i["id_libro"] for i in rec])

    return rec[:N] ,algoritmo_usado
