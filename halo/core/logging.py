"""
Módulo de logging configurable para HALO-S.

Proporciona un sistema de logging integrado que permite controlar
la verbosidad del framework sin configurar logging manualmente.

Uso básico:
    from halo.core.logging import get_logger, set_log_level

    logger = get_logger("halo.training")
    logger.info("Entrenamiento iniciado")

    # Cambiar nivel global
    set_log_level("DEBUG")
"""

import os
import logging
from typing import Optional

# Formato consistente para todos los loggers del framework
_LOG_FORMAT = "[HALO-S] %(levelname)s - %(message)s"

# Registro interno de todos los loggers creados por el framework
_loggers: dict[str, logging.Logger] = {}


def _get_default_level() -> int:
    """
    Obtiene el nivel de logging por defecto.
    Lee la variable de entorno HALO_LOG_LEVEL si está definida,
    de lo contrario retorna INFO.
    """
    env_level = os.environ.get("HALO_LOG_LEVEL", "").upper()
    if env_level:
        # Mapear string a constante de logging
        level = getattr(logging, env_level, None)
        if level is not None:
            return level
    return logging.INFO


def get_logger(name: str = "halo", level: Optional[int] = None) -> logging.Logger:
    """
    Obtiene un logger configurado para HALO-S.

    Si el logger ya fue creado previamente, retorna la misma instancia.
    El formato de salida es consistente: "[HALO-S] %(levelname)s - %(message)s"

    Args:
        name: Nombre del logger (default: "halo").
              Se recomienda usar nombres jerárquicos como "halo.training".
        level: Nivel de logging explícito. Si es None, se usa el valor de
               la variable de entorno HALO_LOG_LEVEL o INFO por defecto.

    Returns:
        Logger de Python configurado con formato consistente.
    """
    # Si ya existe, retornar la misma instancia
    if name in _loggers:
        logger = _loggers[name]
        # Actualizar nivel si se proporciona uno explícito
        if level is not None:
            logger.setLevel(level)
        return logger

    # Crear nuevo logger
    logger = logging.getLogger(name)

    # Determinar nivel: explícito > variable de entorno > INFO
    effective_level = level if level is not None else _get_default_level()
    logger.setLevel(effective_level)

    # Evitar duplicar handlers si el logger ya tiene uno (por re-imports)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setLevel(effective_level)
        formatter = logging.Formatter(_LOG_FORMAT)
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    # No propagar al logger raíz para evitar mensajes duplicados
    logger.propagate = False

    # Registrar para control global
    _loggers[name] = logger
    return logger


def set_log_level(level) -> None:
    """
    Establece el nivel de logging global para todos los loggers de HALO-S.

    Actualiza tanto los loggers existentes como el nivel por defecto
    para loggers que se creen en el futuro.

    Args:
        level: Nivel de logging. Puede ser un entero (logging.DEBUG, etc.)
               o un string ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL").
    """
    # Convertir string a constante numérica si es necesario
    if isinstance(level, str):
        numeric_level = getattr(logging, level.upper(), None)
        if numeric_level is None:
            raise ValueError(
                f"Nivel de logging inválido: '{level}'. "
                f"Valores válidos: DEBUG, INFO, WARNING, ERROR, CRITICAL"
            )
        level = numeric_level

    # Actualizar todos los loggers registrados del framework
    for logger in _loggers.values():
        logger.setLevel(level)
        for handler in logger.handlers:
            handler.setLevel(level)
