"""In-process web jobs. Durable results live in Store, never in Streamlit state."""
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor

from runner import run_evaluation


class JobManager:
    def __init__(self, max_jobs=2):
        self._pool = ThreadPoolExecutor(max_workers=max_jobs, thread_name_prefix='eval-web')
        self._jobs = {}
        self._lock = threading.Lock()

    def start(self, store, suite, model, **kwargs):
        run_id = uuid.uuid4().hex
        event = threading.Event()
        with self._lock:
            # Completed Future results contain only sanitized run snapshots.
            for old_id, job in list(self._jobs.items()):
                if job['future'].done():
                    del self._jobs[old_id]
            future = self._pool.submit(
                run_evaluation, suite, model, on_update=store.save_run,
                cancel_event=event, run_id=run_id, **kwargs
            )
            self._jobs[run_id] = {'future': future, 'event': event}
        return run_id

    def status(self, run_id):
        with self._lock:
            job = self._jobs.get(run_id)
        if job is None:
            return {'known': False, 'done': True, 'error': None}
        done = job['future'].done()
        error = None
        if done:
            try:
                job['future'].result()
            except Exception as exc:
                # The exception text may contain credentials or endpoint details.
                error = type(exc).__name__
        return {'known': True, 'done': done, 'error': error}

    def cancel(self, run_id):
        with self._lock:
            job = self._jobs.get(run_id)
            if job:
                job['event'].set()
                return True
        return False
