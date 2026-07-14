"""Persistencia S3 atómica y verificable para Financial-GPT.

Un prefijo S3 no dispone de rename atómico. Este módulo implementa el patrón
commit-marker:

1. subir todos los archivos del run;
2. subir ``artifact_checksums.json``;
3. verificar tamaños remotos;
4. escribir ``_SUCCESS`` como commit del run;
5. actualizar ``latest.json`` solamente después del commit.

Los loaders se niegan a leer prefijos sin ``_SUCCESS`` y verifican SHA-256 de
los archivos descargados. Así una ejecución interrumpida nunca se presenta como
un modelo válido.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
from typing import Any, Final, Mapping, MutableMapping, Sequence
from urllib.parse import urlparse


DEFAULT_FINANCIAL_GPT_S3_ROOT: Final[str] = (
    "s3://your-private-bucket/users/your-user/financial_gpt"
)
S3_RUN_SCHEMA_VERSION: Final[str] = "1.0"
CHECKSUMS_FILENAME: Final[str] = "artifact_checksums.json"
SUCCESS_FILENAME: Final[str] = "_SUCCESS"
LATEST_FILENAME: Final[str] = "latest.json"
_SAFE_COMPONENT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


@dataclass(frozen=True)
class S3Location:
    """URI S3 normalizada."""

    bucket: str
    key: str = ""

    @classmethod
    def parse(cls, uri: str, *, require_key: bool = False) -> "S3Location":
        parsed = urlparse(str(uri).strip())
        key = parsed.path.lstrip("/").rstrip("/")
        if parsed.scheme != "s3" or not parsed.netloc:
            raise ValueError(f"Invalid S3 URI: {uri!r}")
        if require_key and not key:
            raise ValueError(f"S3 URI must include a key: {uri!r}")
        return cls(bucket=parsed.netloc, key=key)

    @property
    def uri(self) -> str:
        return f"s3://{self.bucket}/{self.key}" if self.key else f"s3://{self.bucket}"

    def join(self, *parts: str) -> "S3Location":
        clean = [self.key] if self.key else []
        for part in parts:
            value = str(part).strip().strip("/")
            if value:
                clean.append(value)
        return S3Location(self.bucket, "/".join(clean))


def validate_component(value: str, *, label: str) -> str:
    normalized = str(value).strip()
    if not _SAFE_COMPONENT.fullmatch(normalized):
        raise ValueError(
            f"{label} must match {_SAFE_COMPONENT.pattern!r}; got {value!r}"
        )
    return normalized


def build_run_uri(
    s3_root: str,
    architecture: str,
    run_id: str,
) -> str:
    """Construye ``<root>/<architecture>/runs/<run_id>``."""

    root = S3Location.parse(s3_root, require_key=True)
    arch = validate_component(architecture, label="architecture").lower()
    run = validate_component(run_id, label="run_id")
    return root.join(arch, "runs", run).uri


def architecture_root_uri(s3_root: str, architecture: str) -> str:
    root = S3Location.parse(s3_root, require_key=True)
    arch = validate_component(architecture, label="architecture").lower()
    return root.join(arch).uri


def assert_run_belongs_to_root(
    run_uri: str,
    *,
    s3_root: str,
    architecture: str,
) -> None:
    run = S3Location.parse(run_uri, require_key=True)
    expected = S3Location.parse(
        architecture_root_uri(s3_root, architecture),
        require_key=True,
    ).join("runs")
    prefix = expected.key + "/"
    if run.bucket != expected.bucket or not run.key.startswith(prefix):
        raise ValueError(
            "run_uri must be under "
            f"{expected.uri}/; got {run_uri!r}"
        )
    relative = run.key[len(prefix) :]
    if "/" in relative or not relative:
        raise ValueError("run_uri must identify exactly one run_id")
    validate_component(relative, label="run_id")


def default_s3_client():
    """Construye boto3 sólo cuando realmente se usa S3."""

    import boto3

    return boto3.client("s3")


@dataclass(frozen=True)
class S3SaveResult:
    run_uri: str
    success_uri: str
    latest_uri: str | None
    file_count: int
    total_bytes: int
    checksums_sha256: str


@dataclass(frozen=True)
class S3DownloadResult:
    local_root: Path
    run_uri: str
    success: Mapping[str, Any]
    checksums: Mapping[str, Any]


def upload_atomic_run(
    local_root: str | Path,
    run_uri: str,
    *,
    s3_root: str,
    architecture: str,
    run_id: str,
    state_digest: str,
    client=None,
    update_latest: bool = True,
) -> S3SaveResult:
    """Sube un directorio y publica el marker sólo al finalizar correctamente."""

    source = Path(local_root).expanduser().resolve()
    if not source.is_dir():
        raise FileNotFoundError(f"local_root does not exist: {source}")
    assert_run_belongs_to_root(
        run_uri,
        s3_root=s3_root,
        architecture=architecture,
    )
    validate_component(run_id, label="run_id")
    destination = S3Location.parse(run_uri, require_key=True)
    api = client or default_s3_client()
    success_key = destination.join(SUCCESS_FILENAME).key
    if _object_exists(api, destination.bucket, success_key):
        raise FileExistsError(f"Committed S3 run already exists: {run_uri}")

    checksums = _build_checksums(source)
    checksums_path = source / CHECKSUMS_FILENAME
    _write_json(checksums_path, checksums)
    checksums_sha256 = _sha256_file(checksums_path)

    uploaded_keys: list[str] = []
    try:
        files = sorted(path for path in source.rglob("*") if path.is_file())
        for path in files:
            relative = path.relative_to(source).as_posix()
            if relative == SUCCESS_FILENAME:
                continue
            key = destination.join(relative).key
            api.upload_file(str(path), destination.bucket, key)
            remote = api.head_object(Bucket=destination.bucket, Key=key)
            expected_size = int(path.stat().st_size)
            if int(remote.get("ContentLength", -1)) != expected_size:
                raise IOError(f"S3 size verification failed for {relative}")
            uploaded_keys.append(key)

        success_payload: MutableMapping[str, Any] = {
            "schema_version": S3_RUN_SCHEMA_VERSION,
            "committed_at_utc": datetime.now(timezone.utc).isoformat(),
            "run_uri": destination.uri,
            "architecture": str(architecture).lower(),
            "run_id": str(run_id),
            "state_digest": str(state_digest),
            "checksums_filename": CHECKSUMS_FILENAME,
            "checksums_sha256": checksums_sha256,
            "file_count": len(checksums["files"]),
            "total_bytes": int(
                sum(int(item["size_bytes"]) for item in checksums["files"].values())
            ),
        }
        _put_json(api, destination.bucket, success_key, success_payload)
        uploaded_keys.append(success_key)
        if not _object_exists(api, destination.bucket, success_key):
            raise IOError("S3 success marker was not committed")

        latest_uri: str | None = None
        if update_latest:
            architecture_root = S3Location.parse(
                architecture_root_uri(s3_root, architecture),
                require_key=True,
            )
            latest = {
                "schema_version": S3_RUN_SCHEMA_VERSION,
                "updated_at_utc": datetime.now(timezone.utc).isoformat(),
                "architecture": str(architecture).lower(),
                "run_id": str(run_id),
                "run_uri": destination.uri,
                "success_uri": destination.join(SUCCESS_FILENAME).uri,
                "state_digest": str(state_digest),
            }
            latest_key = architecture_root.join(LATEST_FILENAME).key
            _put_json(api, destination.bucket, latest_key, latest)
            latest_uri = architecture_root.join(LATEST_FILENAME).uri

        return S3SaveResult(
            run_uri=destination.uri,
            success_uri=destination.join(SUCCESS_FILENAME).uri,
            latest_uri=latest_uri,
            file_count=int(success_payload["file_count"]),
            total_bytes=int(success_payload["total_bytes"]),
            checksums_sha256=checksums_sha256,
        )
    except Exception:
        # Un prefijo incompleto no es visible para loaders porque no contiene
        # _SUCCESS. Limpiamos best-effort para no acumular basura.
        _delete_keys_best_effort(api, destination.bucket, uploaded_keys)
        raise


def resolve_latest_run_uri(
    s3_root: str,
    architecture: str,
    *,
    client=None,
) -> str:
    api = client or default_s3_client()
    root = S3Location.parse(
        architecture_root_uri(s3_root, architecture),
        require_key=True,
    )
    payload = _get_json(api, root.bucket, root.join(LATEST_FILENAME).key)
    run_uri = str(payload.get("run_uri", ""))
    assert_run_belongs_to_root(
        run_uri,
        s3_root=s3_root,
        architecture=architecture,
    )
    return run_uri


def download_verified_run(
    run_uri: str,
    destination: str | Path,
    *,
    client=None,
    include_prefixes: Sequence[str] = ("model/", "evidence/"),
) -> S3DownloadResult:
    """Descarga y verifica modelo/evidencia; evita bajar reportes pesados."""

    source = S3Location.parse(run_uri, require_key=True)
    api = client or default_s3_client()
    success = _get_json(api, source.bucket, source.join(SUCCESS_FILENAME).key)
    if str(success.get("run_uri")) != source.uri:
        raise ValueError("S3 success marker does not match requested run_uri")
    checksums_bytes = _get_bytes(
        api,
        source.bucket,
        source.join(CHECKSUMS_FILENAME).key,
    )
    observed_checksums_hash = hashlib.sha256(checksums_bytes).hexdigest()
    if observed_checksums_hash != str(success.get("checksums_sha256", "")):
        raise ValueError("S3 checksums manifest digest mismatch")
    checksums = json.loads(checksums_bytes.decode("utf-8"))
    if checksums.get("schema_version") != S3_RUN_SCHEMA_VERSION:
        raise ValueError("Unsupported S3 run schema version")

    local_root = Path(destination).expanduser().resolve()
    local_root.mkdir(parents=True, exist_ok=True)
    selected_prefixes = tuple(str(value) for value in include_prefixes)
    files = checksums.get("files")
    if not isinstance(files, Mapping) or not files:
        raise ValueError("S3 checksums manifest has no files")

    for relative, expected in sorted(files.items()):
        safe_relative = _safe_relative_path(str(relative))
        key = source.join(safe_relative).key
        should_download = any(
            safe_relative == prefix.rstrip("/")
            or safe_relative.startswith(prefix.rstrip("/") + "/")
            for prefix in selected_prefixes
        )
        if should_download:
            destination_path = local_root / safe_relative
            destination_path.parent.mkdir(parents=True, exist_ok=True)
            api.download_file(source.bucket, key, str(destination_path))
            if int(destination_path.stat().st_size) != int(expected["size_bytes"]):
                raise ValueError(f"S3 artifact size mismatch: {safe_relative}")
            if _sha256_file(destination_path) != str(expected["sha256"]):
                raise ValueError(f"S3 artifact checksum mismatch: {safe_relative}")
        else:
            remote = api.head_object(Bucket=source.bucket, Key=key)
            if int(remote.get("ContentLength", -1)) != int(expected["size_bytes"]):
                raise ValueError(f"S3 artifact size mismatch: {safe_relative}")

    (local_root / CHECKSUMS_FILENAME).write_bytes(checksums_bytes)
    _write_json(local_root / SUCCESS_FILENAME, success)
    return S3DownloadResult(
        local_root=local_root,
        run_uri=source.uri,
        success=success,
        checksums=checksums,
    )


def _build_checksums(root: Path) -> Mapping[str, Any]:
    files: MutableMapping[str, Mapping[str, Any]] = {}
    for path in sorted(value for value in root.rglob("*") if value.is_file()):
        relative = path.relative_to(root).as_posix()
        if relative in {CHECKSUMS_FILENAME, SUCCESS_FILENAME}:
            continue
        files[relative] = {
            "sha256": _sha256_file(path),
            "size_bytes": int(path.stat().st_size),
        }
    if not files:
        raise ValueError("Cannot persist an empty run directory")
    return {
        "schema_version": S3_RUN_SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "files": files,
    }


def _safe_relative_path(value: str) -> str:
    path = PurePosixPath(value)
    if path.is_absolute() or not path.parts or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(f"Unsafe artifact path in checksums manifest: {value!r}")
    return path.as_posix()


def _put_json(client, bucket: str, key: str, payload: Mapping[str, Any]) -> None:
    body = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    client.put_object(
        Bucket=bucket,
        Key=key,
        Body=body,
        ContentType="application/json",
    )


def _get_json(client, bucket: str, key: str) -> Mapping[str, Any]:
    try:
        raw = _get_bytes(client, bucket, key)
    except Exception as exc:
        if _is_not_found_error(exc):
            raise FileNotFoundError(
                f"Required S3 object not found: s3://{bucket}/{key}"
            ) from exc
        raise
    payload = json.loads(raw.decode("utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"Expected JSON object at s3://{bucket}/{key}")
    return payload


def _get_bytes(client, bucket: str, key: str) -> bytes:
    response = client.get_object(Bucket=bucket, Key=key)
    body = response["Body"]
    return body.read() if hasattr(body, "read") else bytes(body)


def _object_exists(client, bucket: str, key: str) -> bool:
    try:
        client.head_object(Bucket=bucket, Key=key)
        return True
    except Exception as exc:
        if _is_not_found_error(exc):
            return False
        raise


def _is_not_found_error(exc: Exception) -> bool:
    if isinstance(exc, (KeyError, FileNotFoundError)):
        return True
    response = getattr(exc, "response", None)
    if isinstance(response, Mapping):
        error = response.get("Error", {})
        code = str(error.get("Code", "")) if isinstance(error, Mapping) else ""
        return code in {"404", "NoSuchKey", "NotFound"}
    return False


def _delete_keys_best_effort(client, bucket: str, keys: Sequence[str]) -> None:
    if not keys:
        return
    try:
        client.delete_objects(
            Bucket=bucket,
            Delete={"Objects": [{"Key": key} for key in keys], "Quiet": True},
        )
    except Exception:
        pass


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )
