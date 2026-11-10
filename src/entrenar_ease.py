import pandas as pd
import numpy as np
import pickle
import os
from scipy.sparse import csr_matrix
from src.config import OUTPUTS_EASE  

class EASE_Recommender:
    def __init__(self, lambda_reg=500):
        self.lambda_reg = lambda_reg
        self.W = None
        self.user_mapper = {}
        self.reverse_item_mapper = {}
        self.X = None
        
    def fit(self, df, col_user='id_lector', col_item='id_libro', col_rating='rating'):
        print("1. Mapeando IDs a índices...")
        users = df[col_user].astype('category')
        items = df[col_item].astype('category')
        
        self.user_mapper = {user: i for i, user in enumerate(users.cat.categories)}
        self.reverse_item_mapper = {i: item for i, item in enumerate(items.cat.categories)}
        
        row = users.cat.codes.values
        col = items.cat.codes.values
        
        # Opcional: Si el rating tiene nulos o es irrelevante, puedes usar np.ones(len(df))
        data = df[col_rating].astype(np.float32).values
        
        print("2. Creando matriz dispersa (Sparse Matrix)...")
        self.X = csr_matrix((data, (row, col)), shape=(len(self.user_mapper), len(self.reverse_item_mapper)))
        
        print("3. Calculando Matriz de Covarianza (G = X^T * X)...")
        # .toarray() puede consumir mucha RAM si hay > 30,000 libros únicos
        G = self.X.T.dot(self.X).toarray() 
        
        print("4. Sumando penalización y calculando Inversa...")
        diag_indices = np.diag_indices_from(G)
        G[diag_indices] += self.lambda_reg
        P = np.linalg.inv(G) 
        
        print("5. Derivando pesos finales (W)...")
        self.W = P / -np.diag(P)
        np.fill_diagonal(self.W, 0)
        
        print("¡Entrenamiento EASE completado con éxito!")

    def predecir_usuario_con_scores(self, id_lector, top_n=10):
        if id_lector not in self.user_mapper:
            return {}
            
        u_idx = self.user_mapper[id_lector]
        x_u = self.X[u_idx].toarray() 
        
        scores = x_u.dot(self.W).flatten()
        
        leidos_idx = np.where(x_u.flatten() > 0)[0]
        scores[leidos_idx] = -np.inf
        
        top_indices = np.argsort(scores)[::-1][:top_n]
        top_indices = [idx for idx in top_indices if scores[idx] != -np.inf]
        
        # Devuelve diccionario: {id_libro: score_ease}
        return {self.reverse_item_mapper[idx]: scores[idx] for idx in top_indices}

# =====================================================================
# FUNCIÓN PRINCIPAL PARA GESTIONAR LA CACHÉ/ENTRENAMIENTO
# =====================================================================

def obtener_modelo_ease(df_interacciones, state="entrenamiento", lambda_reg=500,min_lecturas=2):
    """
    Verifica si el modelo EASE ya está entrenado y guardado para el estado solicitado.
    Si no existe, lo entrena con el DataFrame proporcionado y lo guarda en disco.
    
    state: "entrenamiento" (usa df_train) o "entrenamiento_final" (usa df_total)
    """
    os.makedirs(OUTPUTS_EASE, exist_ok=True)
    archivo_modelo = os.path.join(OUTPUTS_EASE, f"modelo_ease_{state}.pkl")
    
    if os.path.exists(archivo_modelo):
        print(f"-> Cargando modelo EASE desde caché: {archivo_modelo}")
        with open(archivo_modelo, 'rb') as file:
            modelo = pickle.load(file)
        return modelo
    else:
        print(f"-> Modelo EASE para '{state}' no encontrado. Iniciando entrenamiento...")
        modelo = EASE_Recommender(lambda_reg=lambda_reg)
        
        # Es crucial eliminar Nulos en el rating antes de meterlos a la matriz
        df_limpio = df_interacciones.dropna(subset=['rating']).copy()
        # --- NUEVO: FILTRO PARA SALVAR LA MEMORIA RAM ---
        print(f"Libros originales: {df_limpio['id_libro'].nunique()}")
        conteo = df_limpio['id_libro'].value_counts()
        libros_frecuentes = conteo[conteo >= min_lecturas].index
        df_limpio = df_limpio[df_limpio['id_libro'].isin(libros_frecuentes)]
        print(f"Libros tras filtrar raros (min {min_lecturas} lecturas): {df_limpio['id_libro'].nunique()}")
        # ------------------------------------------------
        
        modelo.fit(df_limpio)
        
        # Guardamos el objeto completo (incluye la matriz W y los mapeos)
        print(f"-> Guardando modelo en: {archivo_modelo}")
        with open(archivo_modelo, 'wb') as file:
            pickle.dump(modelo, file)
            
        return modelo