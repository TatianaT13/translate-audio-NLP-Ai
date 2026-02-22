import json
import time
import argparse
from pathlib import Path
from datetime import datetime

from flash_nlp.transcription import WhisperService
from flash_nlp.io import ensure_dir, load_json, save_json, list_audio_files


def utc_now():
    return datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data/flash_audio_archive", help="Dossier racine contenant les audios")
    ap.add_argument("--output", default="outputs/transcripts_whisper", help="Dossier de sortie")
    ap.add_argument("--model", default="large-v3", help="tiny|base|small|medium|large-v3")
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"], help="cpu ou cuda")
    ap.add_argument("--compute-type", default=None, help="int8|int8_float16|float16|float32 (optionnel)")
    ap.add_argument("--language", default=None, help="ex: fr, en, de. Laisser vide pour auto-detect")
    ap.add_argument("--beam-size", type=int, default=5)
    ap.add_argument("--force", action="store_true", help="Retranscrire même si sortie existe")
    args = ap.parse_args()

    input_dir = Path(args.input)
    out_dir = Path(args.output)
    ensure_dir(out_dir)

    index_path = out_dir / "index.json"
    index = load_json(index_path)

    compute_type = args.compute_type
    if compute_type is None:
        compute_type = "int8" if args.device == "cpu" else "float16"

    print(f"{utc_now()} | Chargement modèle={args.model} device={args.device} compute_type={compute_type}")
    whisper = WhisperService()
    whisper.load(args.model, device=args.device, compute_type=compute_type)

    n_total = 0
    n_done = 0
    n_skipped = 0

    for audio_path in list_audio_files(input_dir):
        n_total += 1
        rel = str(audio_path.relative_to(input_dir))

        safe_rel = rel.replace("/", "__").replace("\\", "__")
        out_json = out_dir / f"{safe_rel}.json"
        out_txt = out_dir / f"{safe_rel}.txt"

        if not args.force and out_json.exists() and out_txt.exists():
            n_skipped += 1
            continue

        t0 = time.time()
        try:
            res = whisper.transcribe_wav_with_segments(
                str(audio_path), args.language, args.beam_size, min_silence_ms=500
            )
            elapsed = time.time() - t0

            meta = {
                "file": rel,
                "abs_path": str(audio_path),
                "created_utc": utc_now(),
                "model": args.model,
                "device": args.device,
                "compute_type": compute_type,
                "elapsed_s": round(elapsed, 3),
                "result": res,
            }

            out_json.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
            out_txt.write_text(res["text"], encoding="utf-8")

            index[rel] = {
                "output_json": str(out_json.relative_to(out_dir)),
                "output_txt": str(out_txt.relative_to(out_dir)),
                "model": args.model,
                "language": res["language"],
                "duration": res["duration"],
                "elapsed_s": round(elapsed, 3),
                "updated_utc": utc_now(),
            }

            n_done += 1
            print(f"{utc_now()} | OK | {rel} | {elapsed:.2f}s | lang={res['language']}")

        except Exception as e:
            print(f"{utc_now()} | ERREUR | {rel} | {e}")

        save_json(index_path, index)

    print(f"{utc_now()} | Terminé | total={n_total} done={n_done} skipped={n_skipped}")
    save_json(index_path, index)


if __name__ == "__main__":
    main()
    