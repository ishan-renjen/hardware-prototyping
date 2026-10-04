import logging
import os
import pathlib

import jax._src.xla_bridge as xb

logger = logging.getLogger(__name__)

_LIB_NAME = "pjrt_c_api_prototype_plugin.so"

def _library_path() -> pathlib.Path:
  override = os.environ.get("PROTOTYPE_PJRT_PLUGIN_PATH")
  if override:
    return pathlib.Path(override)
  # Default: the .so sits next to this file.
  return pathlib.Path(__file__).resolve().parent / _LIB_NAME

def initialize():
  path = _library_path()
  if not path.exists():
    logger.warning("Prototype PJRT plugin not found at %s", path)
    return
  xb.register_plugin("prototype", priority=-1, library_path=str(path))