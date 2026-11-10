import numpy as np
from sklearn.metrics import ndcg_score

def calcular_recall(candidatos_recuperados, true_relevance_dict):
    """
    Calcula el Recall: ¿Qué porcentaje de los libros que el usuario realmente leyó (en test) 
    logramos atrapar en nuestra lista de candidatos?
    """
    libros_test_reales = set(true_relevance_dict.keys())
    
    if len(libros_test_reales) == 0:
        return 0.0
        
    hits = len(set(candidatos_recuperados).intersection(libros_test_reales))
    return hits / len(libros_test_reales)


def calcular_ndcg(true_relevance_dict, predicted_scores_dict, k=10):
    """
    Calcula el NDCG@K aplicando las correcciones contra sesgos y empates.
    """
    # Si el retrieval no devolvió nada, el NDCG es 0 automático
    if len(predicted_scores_dict) == 0:
        return 0.0
        
    # 1. ORDENAMOS ALFABÉTICAMENTE para destruir el sesgo de la concatenación
    id_libros_totales = sorted(list(set(list(true_relevance_dict.keys()) + list(predicted_scores_dict.keys()))))
    
    if len(id_libros_totales) == 0:
        return 0.0
        
    # 2. Armamos el array de relevancia real (0 si el modelo predijo un libro que el usuario no leyó)
    y_true = np.asarray([[true_relevance_dict.get(id_libro, 0) for id_libro in id_libros_totales]])
    
    # 3. Armamos el array de scores (El -99999 manda a los libros no recuperados al fondo)
    y_score_base = np.asarray([[predicted_scores_dict.get(id_libro, -99999) for id_libro in id_libros_totales]])
    
    # 4. ROMPEMOS EMPATES: Sumamos un ruido aleatorio invisible (ej: 0.000004)
    y_score = y_score_base + np.random.uniform(0, 1e-5, size=y_score_base.shape)
    
    # 5. Calculamos la métrica
    return ndcg_score(y_true, y_score, k=k)




import numpy as np
from sklearn.metrics import ndcg_score

def calcular_ndcg_estricto(true_relevance_dict, predicted_scores_dict, k=10):
    """
    Calcula el NDCG@K castigando al modelo si trae menos de K elementos.
    Usa 'dummies' para empujar a los libros reales no recuperados fuera del Top K.
    """
    # Si no trajo absolutamente nada, el castigo es máximo: 0.0
    if len(predicted_scores_dict) == 0:
        return 0.0
        
    # 1. Unimos todos los libros involucrados
    libros_reales = set(true_relevance_dict.keys())
    libros_predichos = set(predicted_scores_dict.keys())
    id_libros_totales = list(libros_reales.union(libros_predichos))
    
    # 2. INYECCIÓN DE DUMMIES: Agregamos K elementos "falsos"
    dummies = [f"dummy_castigo_{i}" for i in range(k)]
    id_libros_totales.extend(dummies)
    
    # 3. Array Y_TRUE: Relevancia real (los dummies obviamente tienen 0)
    y_true = np.asarray([[true_relevance_dict.get(id, 0) for id in id_libros_totales]])
    
    # 4. Array Y_SCORE: Asignación estratégica de puntajes
    y_score_base = []
    for id_libro in id_libros_totales:
        if id_libro in predicted_scores_dict:
            # El modelo lo predijo, conserva su score original
            y_score_base.append(predicted_scores_dict[id_libro])
        elif isinstance(id_libro, str) and id_libro.startswith("dummy_castigo_"):
            # Los dummies reciben un score muy bajo, pero MAYOR que los no recuperados
            y_score_base.append(-50000) 
        else:
            # Los libros reales que el modelo NO ENCONTRÓ reciben el peor score posible.
            # Quedarán detrás de los dummies, fuera del Top K evaluado.
            y_score_base.append(-99999) 
            
    y_score_base = np.asarray([y_score_base])
    
    # 5. Desempate aleatorio microscópico
    y_score = y_score_base + np.random.uniform(0, 1e-5, size=y_score_base.shape)
    
    # 6. Cálculo final
    return ndcg_score(y_true, y_score, k=k)