from __future__ import annotations

import logging
import os
from typing import Optional


def setup_logging(
    log_file: Optional[str] = None,
    level: int = logging.INFO,
) -> None:
    """
    Configura logging básico para el proyecto.

    - Siempre loggea a consola.
    - Opcionalmente loggea también a archivo (log_file).

    Parameters
    ----------
    log_file:
        Ruta al archivo de log. Si es None, sólo se usa consola.
    level:
        Nivel mínimo de logging (logging.INFO, logging.DEBUG, etc.).
    """
    # Evitar configurar dos veces si ya hay handlers
    if logging.getLogger().handlers:
        # Ya está configurado, no hacemos nada
        return

    handlers = [logging.StreamHandler()]

    if log_file is not None:
        log_dir = os.path.dirname(log_file)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, mode="a", encoding="utf-8"))

    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=handlers,
    )


def get_logger(name: str) -> logging.Logger:
    """
    Obtiene un logger con el nombre dado.
    """
    return logging.getLogger(name)