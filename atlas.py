"""Minimal Moth Atlas client with a content-hash job cache.

The key is read from MOTH_API_KEY (stripped: env.bat leaves a trailing space)
and is never logged. Every submitted job is cached under work/jobs/<sha>.json so
re-running an identical request never bills again, and the cache doubles as the
provenance log (engine, params, job_id, timings) for the submission.
"""
import hashlib
import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path

API = "https://api.mothquantum.com/api/v1"
# Cloudflare in front of the API rejects urllib's default UA (error 1010).
# Identify honestly instead of impersonating a browser.
UA = "moth-hack-timesmear/0.1 (python-urllib; hackathon video project)"
DONE = {"completed", "succeeded", "success"}
FAILED = {"failed", "cancelled", "canceled"}


class AtlasError(RuntimeError):
    pass


class Atlas:
    def __init__(self, work: Path = Path(__file__).parent / "work", key: str | None = None):
        key = (key or os.environ.get("MOTH_API_KEY", "")).strip()
        if not key:
            raise AtlasError("MOTH_API_KEY is not set (run env.bat first)")
        self._key = key
        self.jobs_dir = Path(work) / "jobs"
        self.jobs_dir.mkdir(parents=True, exist_ok=True)
        self.assets_dir = Path(work) / "assets"
        self.assets_dir.mkdir(parents=True, exist_ok=True)
        self.network_calls = 0

    # -- low level ---------------------------------------------------------
    def _request(self, method, path, data=None, timeout=120):
        payload = None if data is None else json.dumps(data).encode()
        headers = {"Authorization": f"Bearer {self._key}", "Accept": "application/json",
                   "User-Agent": UA}
        if payload is not None:
            headers["Content-Type"] = "application/json"
        req = urllib.request.Request(API + path, data=payload, headers=headers, method=method)
        self.network_calls += 1
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            detail = e.read(800).decode("utf-8", "replace")
            raise AtlasError(f"{method} {path}: HTTP {e.code}: {detail}") from e

    def me(self):
        return self._request("GET", "/me")

    def upload(self, path: Path, content_type: str) -> str:
        return self.upload_bytes(Path(path).read_bytes(), Path(path).name, content_type)

    def upload_bytes(self, content: bytes, filename: str, content_type: str) -> str:
        """Same as upload(), for data built in memory (e.g. an encoded PNG/WAV) - no temp file.

        Cached by content hash (work/assets/<sha>.json): uploading the SAME bytes twice reuses the
        existing asset id instead of uploading (and paying for) it again. This also fixes job
        caching for any engine driven by uploaded files (blur-v1, retrocausal-echo-v1): a job's
        cache key includes its input_files' asset ids, so a stable, content-derived asset id makes
        an identical job hit Atlas.run()'s own cache too - previously every upload got a fresh
        asset id even for byte-identical content, so those jobs never cached (found the hard way:
        ~20 needlessly repeated blur-v1 jobs while debugging why --spatial looked like it wasn't
        doing anything). Caveat: if Atlas ever garbage-collects an unreferenced asset, a stale
        cached id here would make the downstream job fail loudly (not silently) - delete the
        matching work/assets/<sha>.json to force a re-upload if that happens.
        """
        content_hash = hashlib.sha256(content).hexdigest()
        cache_file = self.assets_dir / f"{content_hash}.json"
        if cache_file.exists():
            return json.loads(cache_file.read_text())["asset_id"]
        asset = self._request("POST", "/assets", {
            "filename": filename, "content_type": content_type, "size_bytes": len(content)})
        up = asset["upload"]
        # Signed headers: send exactly as returned.
        req = urllib.request.Request(up["url"], data=content, headers=up["headers"],
                                     method=up["method"])
        self.network_calls += 1
        with urllib.request.urlopen(req, timeout=300) as r:
            r.read()
        self._request("POST", f"/assets/{asset['asset_id']}/complete")
        cache_file.write_text(json.dumps({"asset_id": asset["asset_id"], "filename": filename}))
        return asset["asset_id"]

    def download(self, url: str) -> bytes:
        self.network_calls += 1
        with urllib.request.urlopen(url, timeout=300) as r:
            return r.read()

    # -- jobs --------------------------------------------------------------
    @staticmethod
    def cache_key(engine: str, params: dict, input_files: dict | None) -> str:
        blob = json.dumps([engine, params, input_files or {}], sort_keys=True,
                          separators=(",", ":")).encode()
        return hashlib.sha256(blob).hexdigest()[:24]

    def run(self, engine: str, params: dict, input_files: dict | None = None,
            poll: float = 2.0, timeout: float = 900, label: str = "") -> dict:
        """Submit (or reuse) a job and return {'job_id', 'result', 'outputs', ...}."""
        sha = self.cache_key(engine, params, input_files)
        record = self.jobs_dir / f"{sha}.json"
        if record.exists():
            saved = json.loads(record.read_text())
            if "result" in saved:
                saved["cached"] = True
                if saved.get("outputs"):
                    # File-output job (blur-v1, retrocausal-echo-v1, ...): the cached "outputs"
                    # hold a PRESIGNED S3 URL with a ~900s TTL from when it was first issued, not
                    # from now - a cache hit days (or even 15+ minutes) later would try to
                    # download an expired URL and get HTTP 403. Re-fetching the result is cheap
                    # and does NOT re-run the (paid) job - it just re-signs a fresh download URL
                    # for the same already-completed job. Hit for real: --echo-audio on the same
                    # clip+params twice, minutes apart, 403'd until this fix.
                    res = self._request("GET", f"/jobs/{saved['job_id']}/result")
                    saved["outputs"] = res.get("outputs") or []
                    if res.get("result") is not None:
                        saved["result"] = res["result"]
                return saved
            job_id = saved["job_id"]  # submitted earlier but never finished
        else:
            body = {"params": params}
            if input_files:
                body["input_files"] = input_files
            accepted = self._request("POST", f"/engines/{engine}/process", body)
            job_id = accepted["job_id"]
            saved = {"engine": engine, "label": label, "job_id": job_id,
                     "submitted_at": accepted.get("submitted_at"),
                     "params_summary": _summarise(params), "input_files": input_files or {}}
            record.write_text(json.dumps(saved, indent=1))
        t0 = time.time()
        deadline = t0 + timeout
        while True:
            status = self._request("GET", f"/jobs/{job_id}/status")
            state = str(status.get("status", "")).lower()
            if state in DONE:
                break
            if state in FAILED:
                # Keep the record as provenance but never reuse a dead job on retry.
                record.replace(record.with_suffix(".failed.json"))
                raise AtlasError(f"{engine} job {job_id} {state}: {status.get('error')}")
            if time.time() > deadline:
                raise AtlasError(f"{engine} job {job_id} still '{state}' after {timeout}s "
                                 f"(id kept in {record.name})")
            time.sleep(poll)
        res = self._request("GET", f"/jobs/{job_id}/result")
        saved.update(result=res.get("result"), outputs=res.get("outputs") or [],
                     seconds=round(time.time() - t0, 2), cached=False)
        record.write_text(json.dumps(saved))
        return saved


def find_output(job: dict, slot: str = "result") -> dict:
    """Pick a file output from a completed job's ``outputs`` list by slot name (falls back to
    the first output if no exact match - most file-output engines only produce one). Used by
    engines that return a file (blur-v1, retrocausal-echo-v1) rather than inline JSON."""
    outputs = job.get("outputs") or []
    if not outputs:
        raise AtlasError(f"job {job.get('job_id')} produced no file outputs")
    return next((o for o in outputs if o.get("slot") == slot), outputs[0])


def _summarise(params: dict) -> dict:
    """Keep provenance small: replace big arrays by their shape."""
    out = {}
    for k, v in params.items():
        if isinstance(v, list) and len(json.dumps(v)) > 200:
            shape, x = [], v
            while isinstance(x, list):
                shape.append(len(x))
                x = x[0] if x else None
            out[k] = f"<array shape={shape}>"
        else:
            out[k] = v
    return out
