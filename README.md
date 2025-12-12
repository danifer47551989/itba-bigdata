# Trabajo Final - Herramientas Para la Extracción y Análisis de Grandes Volumenes de Datos

## Análisis de Sentimientos con Databricks, MLflow y GitHub Actions

Este repositorio contiene el desarrollo del Trabajo Práctico Final de la materia Herramientas para Grandes Volúmenes de Datos, cuyo objetivo principal es integrar conceptos de Big Data, MLOps y aprendizaje automático utilizando Databricks, MLflow, y GitHub Actions para CI/CD.

El proyecto implementa un pipeline completo de análisis de sentimientos basado en un conjunto grande de reseñas textuales, abordando desde la ingesta de datos y su preprocesamiento en PySpark hasta el entrenamiento comparativo de modelos clásicos de Machine Learning optimizados para texto vectorizado.

## Objetivos

1.	Familiarizarse con Git + GitHub para control de versiones y trabajo colaborativo.
2.	Implementar experimentación reproducible con MLflow en Databricks, registrando parámetros, métricas y artefactos.
3.	Diseñar un pipeline reproducible y automatizar pruebas/despliegue usando GitHub Actions.
4.	Evaluar y comparar modelos basados en métricas objetivas.
5.	Registrar y versionar el mejor modelo en MLflow Model Registry con documentación y tags.

## Descripción del Dataset

El dataset elegido consiste en una colección masiva de reseñas textuales con polaridad positiva, negativa o neutral.
Se trabajó inicialmente con un dataset de aproximadamente 15 millones de filas, del cual se extrajo una muestra representativa debido a las limitaciones del entorno Databricks Serverless (free tier).

Preprocesamiento realizado en Databricks
- Limpieza de texto: normalización, remoción de caracteres especiales, eliminación de múltiples espacios.
- Conversión a minúsculas.
- Manejo de valores nulos y textos vacíos.
- Mapeo de etiquetas a formato numérico.
- Tokenización y vectorización mediante TF-IDF (sklearn).
- Split: train/test estratificado.

## Experimentos con MLflow
Se entrenaron múltiples variantes de modelos clásicos optimizados para datos textuales de alta dimensionalidad:

Modelos probados
- Linear SVM (One-vs-Rest)
- Logistic Regression (One-vs-Rest)
- Multinomial Naive Bayes
- Variantes con diferentes hiperparámetros (C, alpha, max_iter, class_weight)

Registrado en MLflow Tracking:
- Parámetros (C, alpha, max_iter, etc.)
- Métricas (precision, recall, F1)
- Artefactos del modelo
- Comparación de variantes

## Selección del Modelo Final
La selección se basó en:

- F1 Macro como métrica principal.
- Robustez del modelo ante clases desbalanceadas.
- Eficiencia dentro de las restricciones de Databricks Serverless.
- Estabilidad del entrenamiento en grandes volúmenes de datos vectorizados.

## CI / GitHub Actions

Agregué varios workflows en la rama `feature/git_actions` para ayudar con el desarrollo y las revisiones:

- `.github/workflows/lint.yml`: comprobaciones de estilo y lint (black, isort, ruff) y ejecución de `pre-commit`.
- `.github/workflows/daily-smoke.yml`: pruebas smoke diarias; ahora también se puede ejecutar manualmente desde la UI.
- `.github/workflows/ci.yml`: CI básico que instala dependencias y ejecuta tests.
- `.github/workflows/codeql.yml`: análisis de seguridad con CodeQL.
- `.github/dependabot.yml`: actualizaciones automáticas de dependencias.
- `requirements-ci.txt`: lista ligera para runs rápidos en CI (evita instalar `pyspark`/`mlflow`).
- `.pre-commit-config.yaml`: hooks para aplicar formato y checks antes de los commits.

Probar localmente (PowerShell)

1) Crear un entorno virtual (recomiendo Python 3.10/3.11) e instalar herramientas ligeras:
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1  # o usar activate.bat en cmd
python -m pip install --upgrade pip
pip install -r requirements-ci.txt
```

2) Comandos útiles:
```powershell
python -m pytest -q
python -m pre_commit run --all-files
python -m black --check .
python -m isort --check-only .
python -m ruff check src test
```


