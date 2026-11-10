"""Environnement des tests gateway — pose AVANT l'import des modules gateway.

gateway.database cree son dossier de donnees a l'import (/app/data dans le
conteneur) : on le redirige vers un dossier temporaire.
"""

import os
import tempfile

os.environ.setdefault("DATA_DIR", tempfile.mkdtemp(prefix="gateway-tests-"))
os.environ.setdefault("GATEWAY_SERVICE_TOKEN", "test-service-token")
os.environ.setdefault("BACKGROUND_TASKS", "false")   # pas de tache de fond pendant les tests
