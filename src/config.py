# DATASETS FILES
BASE_DB = "dataset/recomendador_mangas.db"
TABLA_MODELO_FULL = "interacciones"
TABLA_LIBROS= "libros"
TABLA_LECTORES = "lectores"
EJEMPLO_FILE = "dataset/ejemplo.csv"

# OUTPUTS
       # MODELO FINAL
OUTPUTS_MODELO_TRAIN = "outputs/modelos_train/"
OUTPUTS_MODELO_FINAL = "outputs/modelos_finales/"

OUTPUTS_DATASETS = "outputs/datasets/"

OUTPUTS_RESULTADOS = "outputs/resultados/"

OUTPUTS_CHECKPOINTS ="outputs/checkpoints/"

OUTPUTS_FILTRADO_COLAB = "outputs/filtrado_colaborativo/"

OUTPUTS_EASE = "outputs/ease/"

# FLUJO
#STATES=["entrenamiento", "entrenamiento_final"] o ["test", "prediccion_final"]
# FEATURES
FEATURES_OUT = [
    'id_lector', 
    'id_libro', 
    'nombre',                  # Ojo: en tu SQL cruzado quedó como 'nombre' (no 'nombre_lector')
    'resumen',                 # Texto libre, LightGBM no lo entiende sin NLP
    'titulo',                  # Texto libre
    'fecha_interaccion',       # Fecha cruda
    'vive_en',                 # Lo pasamos acá porque ahora usamos 'pais'
    'rating_promedio_crudo_por_lector' # Lo dejamos afuera a favor de la variable TE (Target Encoded)
]

FEATURES = [
    # --- Features Demográficas / Estáticas (Lector) ---
    'genero',                  # Género del lector
    'nacimiento',              # Año de nacimiento
    'pais',                    # La nueva feature limpia que creamos
    
    # --- Features de Contenido / Estáticas (Libro) ---
    'autor',
    'genero_libro',
    'editorial',
    'anio_edicion',
    
    # --- Features de Comportamiento (Lector) ---
    'volumen_lectura_por_lector',
    'desviacion_estandar_rating_por_lector',
    'genero_favorito_por_lector',
    'ratio_libros_antiguos_vs_nuevos_por_lector',
    'rating_promedio_te_por_lector',
    'pct_autor_favorito_por_lector',
    'pct_genero_favorito_por_lector',
    
    # --- Features de Comportamiento / Popularidad (Libro) ---
    'popularidad_count_por_libro',
    'popularidad_rating_mean_por_libro',
    'rating_promedio_te_por_libro',
    
    # --- Features de Interacción (Candidato vs Historial del Lector) ---
    'pct_lecturas_autor_candidato_por_par_libroylector',
    'pct_lecturas_genero_candidato_por_par_libroylector'
]

CAT_FEATURES = [
    'genero',                     # Género del lector
    'pais',                       # País limpio
    'autor',                      # Autor del libro
    'genero_libro',               # Género del libro
    'editorial',                  # Editorial
    'genero_favorito_por_lector'  # El género más leído por ese usuario
]

#FEATURES_LECTOR=['genero', 'vive_en', 'nacimiento']
#FEATURES_LIBRO=['autor','genero_libro','editorial', 'anio_edicion']
#FEATURES_COMPORTAMIENTO=['volumen_lectura', 'rating_promedio_usuario_te',
 #                        'desviacion_estandar_rating', 'genero_favorito_usuario',
  #                       'cant_antiguos', 'cant_nuevos', 'ratio_libros_antiguos_vs_nuevos']

TARGET='rating'

# SPLIT TRAIN TEST
SEED=42
seeds=[24,50,15,48,59,63,20,15,45,75,85,62,100]

# Lista de países en español
paises_es = [
    "Afganistán", "Albania", "Alemania", "Andorra", "Angola", 
    "Antigua y Barbuda", "Arabia Saudita", "Argelia", "Argentina", "Armenia", 
    "Australia", "Austria", "Azerbaiyán", "Bahamas", "Bangladés", 
    "Barbados", "Baréin", "Bélgica", "Belice", "Benín", 
    "Bielorrusia", "Birmania (Myanmar)", "Bolivia", "Bosnia y Herzegovina", "Botsuana", 
    "Brasil", "Brunéi", "Bulgaria", "Burkina Faso", "Burundi", 
    "Bután", "Cabo Verde", "Camboya", "Camerún", "Canadá", 
    "Catar", "Chad", "Chile", "China", "Chipre", 
    "Ciudad del Vaticano", "Colombia", "Comoras", "Corea del Norte", "Corea del Sur", 
    "Costa de Marfil", "Costa Rica", "Croacia", "Cuba", "Dinamarca", 
    "Dominica", "Ecuador", "Egipto", "El Salvador", "Emiratos Árabes Unidos", 
    "Eritrea", "Eslovaquia", "Eslovenia", "España", "Estados Unidos", 
    "Estonia", "Esuatini (Suazilandia)", "Etiopía", "Filipinas", "Finlandia", 
    "Fiyi", "Francia", "Gabón", "Gambia", "Georgia", 
    "Ghana", "Granada", "Grecia", "Guatemala", "Guinea", 
    "Guinea Ecuatorial", "Guinea-Bisáu", "Guyana", "Haití", "Honduras", 
    "Hungría", "India", "Indonesia", "Irak", "Irán", 
    "Irlanda", "Islandia", "Islas Marshall", "Islas Salomón", "Israel", 
    "Italia", "Jamaica", "Japón", "Jordania", "Kazajistán", 
    "Kenia", "Kirguistán", "Kiribati", "Kuwait", "Laos", 
    "Lesoto", "Letonia", "Líbano", "Liberia", "Libia", 
    "Liechtenstein", "Lituania", "Luxemburgo", "Macedonia del Norte", "Madagascar", 
    "Malasia", "Malaui", "Maldivas", "Malí", "Malta", 
    "Marruecos", "Mauricio", "Mauritania", "México", "Micronesia", 
    "Moldavia", "Mónaco", "Mongolia", "Montenegro", "Mozambique", 
    "Namibia", "Nauru", "Nepal", "Nicaragua", "Níger", 
    "Nigeria", "Noruega", "Nueva Zelanda", "Omán", "Países Bajos", 
    "Pakistán", "Palaos", "Palestina", "Panamá", "Papúa Nueva Guinea", 
    "Paraguay", "Perú", "Polonia", "Portugal", "Reino Unido", 
    "República Centroafricana", "República Checa", "República del Congo", "República Democrática del Congo", "República Dominicana", 
    "Ruanda", "Rumania", "Rusia", "Samoa", "San Cristóbal y Nieves", 
    "San Marino", "San Vicente y las Granadinas", "Santa Lucía", "Santo Tomé y Príncipe", "Senegal", 
    "Serbia", "Seychelles", "Sierra Leona", "Singapur", "Siria", 
    "Somalia", "Sri Lanka", "Sudáfrica", "Sudán", "Sudán del Sur", 
    "Suecia", "Suiza", "Surinam", "Tailandia", "Tanzania", 
    "Tayikistán", "Timor Oriental", "Togo", "Tonga", "Trinidad y Tobago", 
    "Túnez", "Turkmenistán", "Turquía", "Tuvalu", "Ucrania", 
    "Uganda", "Uruguay", "Uzbekistán", "Vanuatu", "Venezuela", 
    "Vietnam", "Yemen", "Yibuti", "Zambia", "Zimbabue"
]

# Lista de países en inglés
countries_en = [
    "Afghanistan", "Albania", "Germany", "Andorra", "Angola", 
    "Antigua and Barbuda", "Saudi Arabia", "Algeria", "Argentina", "Armenia", 
    "Australia", "Austria", "Azerbaijan", "Bahamas", "Bangladesh", 
    "Barbados", "Bahrain", "Belgium", "Belize", "Benin", 
    "Belarus", "Myanmar (Burma)", "Bolivia", "Bosnia and Herzegovina", "Botswana", 
    "Brazil", "Brunei", "Bulgaria", "Burkina Faso", "Burundi", 
    "Bhutan", "Cape Verde", "Cambodia", "Cameroon", "Canada", 
    "Qatar", "Chad", "Chile", "China", "Cyprus", 
    "Vatican City (Holy See)", "Colombia", "Comoros", "North Korea", "South Korea", 
    "Ivory Coast (Côte d'Ivoire)", "Costa Rica", "Croatia", "Cuba", "Denmark", 
    "Dominica", "Ecuador", "Egypt", "El Salvador", "United Arab Emirates", 
    "Eritrea", "Slovakia", "Slovenia", "Spain", "United States", 
    "Estonia", "Eswatini (Swaziland)", "Ethiopia", "Philippines", "Finland", 
    "Fiji", "France", "Gabon", "Gambia", "Georgia", 
    "Ghana", "Grenada", "Greece", "Guatemala", "Guinea", 
    "Equatorial Guinea", "Guinea-Bissau", "Guyana", "Haiti", "Honduras", 
    "Hungary", "India", "Indonesia", "Iraq", "Iran", 
    "Ireland", "Iceland", "Marshall Islands", "Solomon Islands", "Israel", 
    "Italy", "Jamaica", "Japan", "Jordan", "Kazakhstan", 
    "Kenya", "Kyrgyzstan", "Kiribati", "Kuwait", "Laos", 
    "Lesotho", "Latvia", "Lebanon", "Liberia", "Libya", 
    "Liechtenstein", "Lithuania", "Luxembourg", "North Macedonia", "Madagascar", 
    "Malaysia", "Malawi", "Maldives", "Mali", "Malta", 
    "Morocco", "Mauritius", "Mauritania", "Mexico", "Micronesia", 
    "Moldova", "Monaco", "Mongolia", "Montenegro", "Mozambique", 
    "Namibia", "Nauru", "Nepal", "Nicaragua", "Niger", 
    "Nigeria", "Norway", "New Zealand", "Oman", "Netherlands", 
    "Pakistan", "Palau", "Palestine", "Panama", "Papua New Guinea", 
    "Paraguay", "Peru", "Poland", "Portugal", "United Kingdom", 
    "Central African Republic", "Czech Republic (Czechia)", "Republic of the Congo", "Democratic Republic of the Congo", "Dominican Republic", 
    "Rwanda", "Romania", "Russia", "Samoa", "Saint Kitts and Nevis", 
    "San Marino", "Saint Vincent and the Grenadines", "Saint Lucia", "Sao Tome and Principe", "Senegal", 
    "Serbia", "Seychelles", "Sierra Leone", "Singapore", "Syria", 
    "Somalia", "Sri Lanka", "South Africa", "Sudan", "South Sudan", 
    "Sweden", "Switzerland", "Suriname", "Thailand", "Tanzania", 
    "Tajikistan", "East Timor (Timor-Leste)", "Togo", "Tonga", "Trinidad and Tobago", 
    "Tunisia", "Turkmenistan", "Turkey", "Tuvalu", "Ukraine", 
    "Uganda", "Uruguay", "Uzbekistan", "Vanuatu", "Venezuela", 
    "Vietnam", "Yemen", "Djibouti", "Zambia", "Zimbabwe"
]