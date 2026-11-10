"""Limite de taille des fichiers envoyes.

Deux barrieres :
- un middleware refuse la requete sur son en-tete Content-Length, avant que le
  corps ne soit recu (sinon le serveur l'ecrit en entier sur disque) ;
- la lecture par blocs s'arrete des que la limite est depassee, pour les envois
  sans Content-Length : on ne charge jamais plus que la limite en memoire.
"""
import os

from fastapi import HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse

MAX_UPLOAD_MB    = int(os.getenv("MAX_UPLOAD_MB", "100"))
MAX_UPLOAD_BYTES = MAX_UPLOAD_MB * 1024 * 1024
# Marge pour l'enveloppe multipart (champs du formulaire, separateurs)
_BODY_OVERHEAD   = 1024 * 1024
_CHUNK           = 1024 * 1024

TOO_LARGE = f"Fichier trop volumineux (limite {MAX_UPLOAD_MB} Mo). Coupe ton audio en segments plus courts."


async def reject_oversized_body(request: Request, call_next):
    """Middleware HTTP : 413 des l'en-tete si le corps annonce depasse la limite."""
    length = request.headers.get("content-length")
    if length and length.isdigit() and int(length) > MAX_UPLOAD_BYTES + _BODY_OVERHEAD:
        return JSONResponse(status_code=413, content={"detail": TOO_LARGE})
    return await call_next(request)


async def read_capped(file: UploadFile) -> bytes:
    """Contenu du fichier ; 413 au-dela de MAX_UPLOAD_MB."""
    chunks: list[bytes] = []
    size = 0
    while chunk := await file.read(_CHUNK):
        size += len(chunk)
        if size > MAX_UPLOAD_BYTES:
            raise HTTPException(status_code=413, detail=TOO_LARGE)
        chunks.append(chunk)
    return b"".join(chunks)
