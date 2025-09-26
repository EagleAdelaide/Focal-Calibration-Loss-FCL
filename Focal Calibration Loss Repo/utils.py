import os, time, json, random, socket

def set_seed(seed: int = 42):
    import torch
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

def fmt_eta(seconds: float) -> str:
    seconds = max(0, int(seconds + 0.5))
    m, s = divmod(seconds, 60); h, m = divmod(m, 60); d, h = divmod(h, 24)
    if d > 0: return f"{d}d {h:02d}:{m:02d}:{s:02d}"
    if h > 0: return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"

def _safe_key(s: str) -> str:
    return s.replace("|", "_").replace("/", "-").replace(" ", "_")

def lock_path(out_dir: str, run_key: str) -> str:
    return os.path.join(out_dir, f"lock_{_safe_key(run_key)}.lock")

def acquire_lock(out_dir: str, run_key: str, stale_hours: int = 48) -> str | None:
    lp = lock_path(out_dir, run_key)
    flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
    try:
        fd = os.open(lp, flags)
        with os.fdopen(fd, "w") as f:
            f.write(json.dumps({
                "host": socket.gethostname(),
                "pid": os.getpid(),
                "start_ts": time.time(),
                "run_key": run_key
            }, indent=2))
        return lp
    except FileExistsError:
        try:
            mtime = os.path.getmtime(lp)
            if (time.time() - mtime) > stale_hours * 3600:
                print(f"[LOCK] Stale lock detected for {run_key}. Removing {lp}.")
                try: os.remove(lp)
                except FileNotFoundError: pass
                try:
                    fd = os.open(lp, flags)
                    with os.fdopen(fd, "w") as f:
                        f.write(json.dumps({"note": "recovered stale lock", "host": socket.gethostname(),
                                            "pid": os.getpid(), "start_ts": time.time(), "run_key": run_key}, indent=2))
                    return lp
                except FileExistsError:
                    return None
        except FileNotFoundError:
            return None
        return None

def release_lock(lock_file: str | None):
    try:
        if lock_file and os.path.exists(lock_file):
            os.remove(lock_file)
    except Exception as e:
        print(f"[LOCK] Failed to remove lock {lock_file}: {e}")
