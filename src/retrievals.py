import pandas as pd
import sqlite3

from src.config import BASE_DB


def retrieval_q1_populares(id_lector, top_n, cache):
    leidos = cache["dict_leidos"].get(id_lector, set())
    # Iteramos sobre la lista global, frenando cuando llegamos a top_n
    candidatos = []
    for lib in cache["populares_global"]:
        if lib not in leidos:
            candidatos.append(lib)
            if len(candidatos) == top_n: break
    return candidatos

def retrieval_q2_mismo_autor(id_lector, top_n, cache):
    leidos = cache["dict_leidos"].get(id_lector, set())
    if not leidos:
        return []
    
    # Identificamos qué autores leyó usando el set de leídos
    autores_leidos = {cache["dict_libro_autor"].get(lib) for lib in leidos}
    autores_leidos.discard(None) # Por si hay vacíos
    
    # Juntamos todos los libros de esos autores
    candidatos_potenciales = set()
    for autor in autores_leidos:
        candidatos_potenciales.update(cache["libros_por_autor"].get(autor, set()))
        
    # Removemos los que ya leyó y ordenamos por popularidad para devolver los mejores
    candidatos_validos = candidatos_potenciales - leidos
    return [lib for lib in cache["populares_global"] if lib in candidatos_validos][:top_n]

def retrieval_q3_mismo_genero(id_lector, top_n, cache):
    leidos = cache["dict_leidos"].get(id_lector, set())
    if not leidos:
        return []
        
    generos_leidos = {cache["dict_libro_genero"].get(lib) for lib in leidos}
    generos_leidos.discard(None)
    
    candidatos_potenciales = set()
    for genero in generos_leidos:
        candidatos_potenciales.update(cache["libros_por_genero"].get(genero, set()))
        
    candidatos_validos = candidatos_potenciales - leidos
    return [lib for lib in cache["populares_global"] if lib in candidatos_validos][:top_n]

def retrieval_q4_popularidad_pais(id_lector, top_n, cache):
    leidos = cache["dict_leidos"].get(id_lector, set())
    pais = cache["dict_lector_pais"].get(id_lector, "Desconocido")
    
    # Si no conocemos el país, devolvemos vacío
    populares_del_pais = cache["populares_por_pais"].get(pais, [])
    
    candidatos = []
    for lib in populares_del_pais:
        if lib not in leidos:
            candidatos.append(lib)
            if len(candidatos) == top_n: break
    return candidatos

def retrieval_q5_trending(id_lector, top_n, cache):
    leidos = cache["dict_leidos"].get(id_lector, set())
    
    candidatos = []
    for lib in cache["populares_trending"]:
        if lib not in leidos:
            candidatos.append(lib)
            if len(candidatos) == top_n: break
    return candidatos

def retrieval_hibrido(id_lector, top_n, cache):
    # Asignamos una cuota equitativa a cada estrategia
    top_n_por_query = max(1, top_n // 5) 
    
    c1 = retrieval_q1_populares(id_lector, top_n_por_query, cache)
    c2 = retrieval_q2_mismo_autor(id_lector, top_n_por_query, cache)
    c3 = retrieval_q3_mismo_genero(id_lector, top_n_por_query, cache)
    c4 = retrieval_q4_popularidad_pais(id_lector, top_n_por_query, cache)
    c5 = retrieval_q5_trending(id_lector, top_n_por_query, cache)
    
    candidatos_unicos = list(set(c1 + c2 + c3 + c4 + c5))
    return candidatos_unicos


def retrieval_q6_productos_asociados(id_lector, top_n, cache):
    leidos = cache["dict_leidos"].get(id_lector, set())
    if not leidos:
        return []
        
    candidatos_potenciales = set()
    
    # Buscamos todos los vecinos de los libros que leyó
    for lib in leidos:
        asociados = cache["jaccard_matrix"].get(lib, {}).keys()
        candidatos_potenciales.update(asociados)
        
    # Quitamos los que ya leyó
    candidatos_validos = list(candidatos_potenciales - leidos)
    
    return candidatos_validos[:top_n]

def retrieval_q7_ease(id_lector, top_n, cache=None):
    """Retrieval usando el modelo EASE entrenado."""
    modelo_ease = cache.get('modelo_ease')
    
    if modelo_ease is None:
        print("Advertencia: modelo_ease no encontrado en cache.")
        return []
        
    # EASE devuelve un diccionario {id_libro: score}. Solo extraemos las keys.
    dict_scores = modelo_ease.predecir_usuario_con_scores(id_lector, top_n=top_n)
    return list(dict_scores.keys())