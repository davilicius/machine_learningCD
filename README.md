# 🤖 Aprendizaje Supervisado: Validación y Optimización de Modelos

¡Bienvenido al repositorio de **Validación y Ajuste de Modelos de Machine Learning**! Este proyecto tiene como objetivo ilustrar de forma práctica y pedagógica cómo evaluar el rendimiento de un modelo predictivo, diagnosticar problemas de **sobreajuste (overfitting)** y **subajuste (underfitting)**, y optimizar hiperparámetros utilizando un **Árbol de Decisión (`DecisionTreeRegressor`)** sobre el clásico dataset de precios de viviendas de Melbourne (`melb_data.csv`).

---

## 📋 Tabla de Contenidos
- [✨ Características Principales](#-características-principales)
- [🏗️ Estructura del Proyecto](#️-estructura-del-proyecto)
- [🔄 Flujo de Trabajo (Workflow)](#-flujo-de-trabajo-workflow)
- [⚙️ Requisitos e Instalación](#️-requisitos-e-instalación)
- [📘 Resumen de Lecciones](#-resumen-de-lecciones)
  - [Lección 1: Validación de Modelos](#lección-1-validación-de-modelos)
  - [Lección 2: Underfitting y Overfitting](#lección-2-underfitting-y-overfitting)
- [📊 Resultados Obtenidos](#-resultados-obtenidos)
- [🚀 Cómo Ejecutar](#-cómo-ejecutar)

---

## ✨ Características Principales
* **Métricas Claras:** Implementación y explicación matemática y práctica del **Error Absoluto Medio (MAE)**.
* **Validación fuera de muestra:** Demostración del error crítico de validar "dentro de la muestra" (*in-sample*) y cómo solucionarlo con división de entrenamiento/prueba (*train-test split*).
* **Búsqueda del Punto Óptimo:** Control de la complejidad del modelo ajustando el parámetro `max_leaf_nodes` para evitar el ruido y capturar patrones reales.
* **Visualización de Datos:** Gráficos dinámicos con `seaborn` y `matplotlib` para mapear el MAE frente a la cantidad de nodos.

---

## 🏗️ Estructura del Proyecto

```text
machine_learningCD/
├── data/
│   └── melb_data.csv       # Dataset de viviendas de Melbourne (Kaggle)
├── img/
│   ├── mae.png             # Gráfico conceptual de error vs complejidad
│   └── tree3.png           # Diagrama conceptual de divisiones de un árbol
├── 1_validacion_modelos.ipynb         # Lección 1: MAE y división de datos
├── 2_underfitting_y_overfitting.ipynb # Lección 2: Optimización de max_leaf_nodes
├── requirements.txt        # Dependencias del proyecto
└── README.md               # Documentación principal (este archivo)
```

---

## 🔄 Flujo de Trabajo (Workflow)

El siguiente diagrama de flujo ilustra el ciclo completo implementado en los cuadernos para entrenar, validar y optimizar el modelo de regresión:

```mermaid
graph TD
    A["melb_data.csv (Dataset de Melbourne)"] --> B["Limpieza de Datos (dropna)"]
    B --> C["Definir Target (Price) y Features"]
    C --> D["Dividir Dataset (train_test_split)"]
    D --> E["train_X, train_y (Entrenamiento)"]
    D --> F["val_X, val_y (Validación)"]
    E --> G["Entrenar DecisionTreeRegressor"]
    G --> H["Predecir sobre val_X"]
    H --> I["Calcular Error Absoluto Medio (MAE)"]
    F --> I
    I --> J["Evaluar max_leaf_nodes (Tuning)"]
    J --> K{"¿Menor MAE entre candidatos?"}
    K -->|Sí| L["Evaluar en test independiente"]
    K -->|No| G
```

---

## ⚙️ Requisitos e Instalación

Este proyecto requiere **Python 3.10+** y los paquetes listados en `requirements.txt`.

### 1. Clonar o acceder al directorio del proyecto:
```bash
cd machine_learningCD
```

### 2. Crear y activar un entorno virtual:
```bash
# En macOS/Linux
python -m venv .venv
source .venv/bin/activate
```

### 3. Instalar las dependencias necesarias:
```bash
pip install -r requirements.txt
```

> [!NOTE]
> Las dependencias principales instaladas son:
> * `scikit-learn` - Para la creación, división y evaluación del modelo.
> * `pandas` - Para manipulación y limpieza de datos estructurados.
> * `numpy` - Para operaciones numéricas eficientes.
> * `seaborn` & `matplotlib` - Para visualización de métricas y tendencias.
> * `jupyterlab` - Para la visualización y ejecución interactiva de cuadernos.

---

## 📘 Resumen de Lecciones

### Lección 1: Validación de Modelos
Esta sección aborda la pregunta fundamental: **¿Qué tan preciso es nuestro modelo con datos nuevos?**

* **Métrica MAE:** Se introduce el Error Absoluto Medio, definido formalmente como:
  $$\text{MAE} = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i|$$
  *En términos sencillos: "En promedio, nuestras predicciones están equivocadas por X cantidad de dólares".*
* **La trampa del error "In-Sample":** Se demuestra que si evaluamos el modelo con los mismos datos con los que entrenó, el MAE fue 434.72 dentro de la muestra y 265,345.35 en validación: evaluar con los datos de entrenamiento subestima el error fuera de muestra.
* **Solución:** Se implementa `train_test_split` de Scikit-Learn para reservar un **25%** de los datos exclusivamente para validación, asegurando una evaluación honesta y realista.

### Lección 2: Underfitting y Overfitting
En esta lección exploramos cómo controlar la complejidad del árbol de decisión para encontrar el equilibrio perfecto.

* **Sobreajuste (Overfitting):** Ocurre cuando el árbol es demasiado profundo. El modelo memoriza el ruido y las peculiaridades del set de entrenamiento, incrementando el error en los datos de validación.
* **Subajuste (Underfitting):** Ocurre cuando el árbol es demasiado superficial (pocas hojas). El modelo no logra capturar los patrones generales y tiene un rendimiento deficiente tanto en entrenamiento como en validación.
* **Optimización (`max_leaf_nodes`):** Se crea una función iterativa `calcular_mae` para probar múltiples configuraciones de nodos (`[5, 50, 500, 600, 700, 800, 1000, 5000]`), analizando la curva de error resultante.

---

## 📊 Resultados Obtenidos

Al realizar el análisis de validación y optimización de hiperparámetros sobre la base de datos de viviendas de Melbourne con las siguientes columnas predictoras (*Features*):
* `Rooms` (Habitaciones)
* `Bathroom` (Baños)
* `Landsize` (Tamaño del terreno)
* `BuildingArea` (Área de construcción)
* `YearBuilt` (Año de construcción)
* `Lattitude` (Latitud)
* `Longtitude` (Longitud)

Resultados reproducidos el 21 de octubre de 2026 con ejecución asistida por IA. Se ejecutaron 15 celdas de código sin errores; las salidas están guardadas en los notebooks. Se fijó `random_state=0` en la división y los árboles.

De 13,580 filas originales quedaron 6,196 completas: 4,647 para entrenamiento y 1,549 para validación. El MAE del árbol sin límite fue 265,345.35, frente a 483,397.47 de una referencia que siempre predice la mediana de entrenamiento.

| Máximo de hojas | MAE de validación (redondeado) |
| --- | ---: |
| 5 | 347,380 |
| 50 | 258,171 |
| 500 | 243,495 |
| 600 | 243,951 |
| **700** | **242,955** |
| 800 | 244,042 |
| 1000 | 247,378 |
| 5000 | 255,575 |

**Conclusión:** 700 hojas obtuvo el menor MAE entre los candidatos probados en esta partición. Esto no prueba un óptimo universal ni autoriza el despliegue. Tras elegir hiperparámetros se necesita una prueba independiente; el conjunto usado para elegirlos ya no ofrece una estimación final imparcial. Eliminar todas las filas incompletas también puede introducir sesgo.

---

## 🚀 Cómo Ejecutar

Para abrir y experimentar con los cuadernos en tu computadora local:

1. Inicia JupyterLab en tu terminal:
   ```bash
   jupyter lab
   ```
2. En la interfaz web de Jupyter, abre el archivo `1_validacion_modelos.ipynb` y ejecuta las celdas secuencialmente para observar el impacto de la separación de datos de validación.
3. Posteriormente, abre `2_underfitting_y_overfitting.ipynb` para visualizar de manera gráfica cómo se optimiza el modelo de Machine Learning mediante la curva de entrenamiento y validación.

---
*Desarrollado como recurso de aprendizaje interactivo para fundamentos de Ciencia de Datos y Machine Learning.*
