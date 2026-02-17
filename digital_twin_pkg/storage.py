import sqlite3
import json
import os
import time
import uuid
import threading
import contextlib
import shutil
import random
import logging
import copy
from collections import deque
from typing import Any, Dict, List, Optional
from .config import *

logger = logging.getLogger(__name__)

class StorageManager:
    """
    Handles SQLite (WAL) and JSON file persistence with atomic locking and transactions.
    """
    def __init__(self, tenant_id: str, base_data_dir: str):
        self.tenant_id = tenant_id
        # Ensure directory exists
        self.data_dir = os.path.join(base_data_dir, self.tenant_id)
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir, mode=0o700, exist_ok=True)
            
        self.paths = {
            "rules": os.path.join(self.data_dir, "digital_twin_rules.json"),
            "history": os.path.join(self.data_dir, "prediction_history.json"),
            "noise_counts": os.path.join(self.data_dir, "noise_counter.json"),
            "rule_stats": os.path.join(self.data_dir, "rule_stats.json"),
            "metric_store": os.path.join(self.data_dir, "metric_store.json"), 
            "outcomes": os.path.join(self.data_dir, "outcomes.json"),
            "incident_register": os.path.join(self.data_dir, "incident_register.json"),
            "maintenance_windows": os.path.join(self.data_dir, "maintenance_windows.json"),
            "event_log_jsonl": os.path.join(self.data_dir, "event_log.jsonl"),
            "event_log_dir": os.path.join(self.data_dir, "events"),
            "optimization_log": os.path.join(self.data_dir, "optimization_log.json"),
            "evaluation_report": os.path.join(self.data_dir, "evaluation_report.json"),
            "evaluation_state": os.path.join(self.data_dir, "evaluation_state.json"),
            "shadow_eval_state": os.path.join(self.data_dir, "shadow_eval_state.json"),
            "lock_dir": os.path.join(self.data_dir, "lock.dir"),
            "tenant_keyring": os.path.join(self.data_dir, "tenant_keyring.json"),
            "sqlite_db": os.path.join(self.data_dir, "state.db"),
        }
        
        if not os.path.exists(self.paths["event_log_dir"]):
            os.makedirs(self.paths["event_log_dir"], mode=0o700, exist_ok=True)

        self._db_lock = threading.Lock()
        self._conn = None
        self._init_sqlite()

    # --- Lock ---
    @contextlib.contextmanager
    def global_lock(self, timeout_sec: float = LOCK_TIMEOUT_SEC):
        lock_path = self.paths["lock_dir"]
        start_time = time.monotonic()
        got_lock = False
        while True:
            try:
                os.mkdir(lock_path)
                got_lock = True
                break
            except FileExistsError:
                try:
                    if time.time() - os.path.getmtime(lock_path) > LOCK_TTL_SEC:
                        shutil.rmtree(lock_path, ignore_errors=True)
                        continue
                except: pass
                if time.monotonic() - start_time > timeout_sec:
                    raise TimeoutError(f"Could not acquire lock for {self.tenant_id}")
                time.sleep(LOCK_POLL_SEC + random.uniform(0, 0.05))
            except Exception:
                raise
        try:
            yield
        finally:
            if got_lock:
                try:
                    os.rmdir(lock_path)
                except:
                    pass

    # --- SQLite Init & Schema ---
    def _init_sqlite(self):
        try:
            self._conn = sqlite3.connect(self.paths["sqlite_db"], check_same_thread=False)
            self._conn.execute('PRAGMA journal_mode=WAL;')
            with self._db_lock:
                self._conn.execute('CREATE TABLE IF NOT EXISTS state (key TEXT PRIMARY KEY, value TEXT, updated_at REAL)')
                self._conn.execute('CREATE TABLE IF NOT EXISTS metrics (device_id TEXT, rule_pattern TEXT, metric_name TEXT, timestamp REAL, value REAL)')
                self._conn.execute('CREATE INDEX IF NOT EXISTS idx_metrics_query ON metrics (device_id, rule_pattern, metric_name, timestamp)')
                
                self._conn.execute('''
                    CREATE TABLE IF NOT EXISTS audit_log (
                        event_id TEXT PRIMARY KEY, timestamp REAL, event_type TEXT, 
                        actor TEXT, rule_pattern TEXT, details_json TEXT, 
                        status TEXT DEFAULT 'committed', 
                        rules_hash_before TEXT, rules_hash_after TEXT, error TEXT
                    )
                ''')
                
                self._conn.execute('''
                    CREATE TABLE IF NOT EXISTS rule_config (
                        rule_pattern TEXT PRIMARY KEY, 
                        paging_threshold REAL, 
                        logging_threshold REAL, 
                        rule_json TEXT, 
                        updated_at REAL
                    )
                ''')
                self._conn.commit()
        except Exception as e:
            logger.warning(f"SQLite init failed: {e}")
            self._conn = None

    # --- State Persistence ---
    def save_state_sqlite(self, key: str, value: Any):
        if not self._conn: return
        try:
            val_json = json.dumps(value, ensure_ascii=False)
            with self._db_lock:
                self._conn.execute('INSERT OR REPLACE INTO state (key, value, updated_at) VALUES (?, ?, ?)',
                                   (key, val_json, time.time()))
                self._conn.commit()
        except Exception: pass

    def load_state_sqlite(self, key: str, default: Any) -> Any:
        if not self._conn: return default
        try:
            with self._db_lock:
                cur = self._conn.cursor()
                cur.execute('SELECT value FROM state WHERE key = ?', (key,))
                row = cur.fetchone()
            return json.loads(row[0]) if row else default
        except Exception: return default

    # --- JSON Atomic File Ops ---
    def load_json(self, key: str, default: Any = None) -> Any:
        if key in ["evaluation_state"] and self._conn:
            val = self.load_state_sqlite(key, None)
            if val is not None: return val
            
        path = self.paths.get(key)
        if path and os.path.exists(path):
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except: return default
        return default

    def save_json_atomic(self, key: str, data: Any):
        path = self.paths.get(key)
        if not path: return
        temp_path = path + ".tmp." + uuid.uuid4().hex
        try:
            def default_serializer(obj):
                if isinstance(obj, (set, deque)): return list(obj)
                return str(obj)
            with open(temp_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=4, ensure_ascii=False, default=default_serializer)
            os.chmod(temp_path, 0o600)
            os.replace(temp_path, path)
        except Exception:
            # 修正箇所: try-exceptを複数行に分割
            if os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except:
                    pass

    # --- Metrics DB ---
    def db_insert_metric(self, dev_id, rule_ptn, metric_name, ts, val):
        if not self._conn: return
        try:
            with self._db_lock:
                self._conn.execute('INSERT INTO metrics VALUES (?, ?, ?, ?, ?)', 
                                   (dev_id, rule_ptn, metric_name, ts, float(val)))
                self._conn.commit()
        except Exception: pass

    def db_fetch_metrics(self, dev_id, rule_ptn, metric_name, min_ts):
        if not self._conn: return []
        try:
            with self._db_lock:
                cur = self._conn.cursor()
                cur.execute('''SELECT timestamp, value FROM metrics 
                               WHERE device_id=? AND rule_pattern=? AND metric_name=? AND timestamp >= ? 
                               ORDER BY timestamp ASC''', (dev_id, rule_ptn, metric_name, min_ts))
                return cur.fetchall()
        except Exception: return []

    def db_cleanup_metrics(self, retention_sec):
        if not self._conn: return
        try:
            cutoff = time.time() - retention_sec
            with self._db_lock:
                self._conn.execute('DELETE FROM metrics WHERE timestamp < ?', (cutoff,))
                self._conn.commit()
        except Exception: pass

    # --- Rule Config DB ---
    def rule_config_upsert(self, rp, pt, lt, rule_json_str):
        if not self._conn: return False
        try:
            with self._db_lock:
                self._conn.execute(
                    'INSERT OR REPLACE INTO rule_config (rule_pattern, paging_threshold, logging_threshold, rule_json, updated_at) VALUES (?, ?, ?, ?, ?)',
                    (rp, pt, lt, rule_json_str, time.time())
                )
                self._conn.commit()
            return True
        except Exception: return False

    def rule_config_get_json_str(self, rp):
        if not self._conn: return None
        try:
            with self._db_lock:
                cur = self._conn.cursor()
                cur.execute('SELECT rule_json FROM rule_config WHERE rule_pattern = ?', (rp.lower(),))
                row = cur.fetchone()
            return row[0] if row else None
        except: return None
        
    def rule_config_get_all_json_strs(self) -> List[str]:
        if not self._conn: return []
        try:
            with self._db_lock:
                cur = self._conn.cursor()
                cur.execute('SELECT rule_json FROM rule_config ORDER BY rule_pattern ASC')
                rows = cur.fetchall()
            return [r[0] for r in rows if r[0]]
        except: return []

    # --- Audit Log (Transactional) ---
    def audit_insert_prepared(self, event, hash_before):
        if not self._conn: return False
        try:
            d = json.dumps({
                "iso_time": event.get("iso_time"),
                "apply_mode": event.get("apply_mode"),
                "changes": event.get("changes"),
                "evidence": event.get("evidence"),
                "details": event.get("details")
            }, ensure_ascii=False)
            
            with self._db_lock:
                self._conn.execute(
                    "INSERT OR REPLACE INTO audit_log (event_id, timestamp, event_type, actor, rule_pattern, details_json, status, rules_hash_before) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (event["event_id"], float(event["timestamp"]), event["event_type"], event["actor"], event["rule_pattern"], d, "prepared", hash_before)
                )
                self._conn.commit()
            return True
        except Exception: return False

    def audit_mark_committed(self, event_id, hash_after):
        if not self._conn: return False
        try:
            with self._db_lock:
                self._conn.execute("UPDATE audit_log SET status='committed', rules_hash_after=?, error=NULL WHERE event_id=?", 
                                   (hash_after, event_id))
                self._conn.commit()
            return True
        except Exception: return False

    def audit_mark_aborted(self, event_id, error_msg):
        if not self._conn: return False
        try:
            with self._db_lock:
                self._conn.execute("UPDATE audit_log SET status='aborted', error=? WHERE event_id=?", 
                                   ((error_msg or "")[:2000], event_id))
                self._conn.commit()
            return True
        except Exception: return False
        
    def audit_log_generic(self, event: Dict[str, Any]):
        if not self._conn: return
        try:
            details = json.dumps(event.get("details", {}), ensure_ascii=False)
            with self._db_lock:
                self._conn.execute(
                    'INSERT INTO audit_log (event_id, timestamp, event_type, actor, rule_pattern, details_json, status) VALUES (?, ?, ?, ?, ?, ?, ?)',
                    (str(event.get("event_id") or uuid.uuid4()), float(event.get("timestamp") or time.time()), 
                     str(event.get("event_type")), str(event.get("actor")), str(event.get("rule_pattern")), details, "committed")
                )
                self._conn.commit()
        except Exception: pass

    # --- Rule Config Seeding ---
    def _seed_rule_config_from_rules_json(self, rules_data: List[dict]):
        """
        rules.json の内容を rule_config テーブルに初期シードする。
        既存レコードは上書きしない（INSERT OR IGNORE）。
        engine.py の _load_rules() / repair_db_from_rules_json() から呼び出される。
        """
        if not self._conn or not rules_data:
            return
        try:
            with self._db_lock:
                for item in rules_data:
                    rp  = str(item.get("pattern", "")).lower()
                    pt  = float(item.get("paging_threshold")  or 0.40)
                    lt  = float(item.get("logging_threshold") or 0.35)
                    rj  = json.dumps(
                        {k: v for k, v in item.items() if not k.startswith('_')},
                        ensure_ascii=False
                    )
                    if not rp:
                        continue
                    self._conn.execute(
                        "INSERT OR IGNORE INTO rule_config "
                        "(rule_pattern, paging_threshold, logging_threshold, rule_json, updated_at) "
                        "VALUES (?, ?, ?, ?, ?)",
                        (rp, pt, lt, rj, time.time())
                    )
                self._conn.commit()
        except Exception as e:
            logger.warning(f"_seed_rule_config_from_rules_json failed: {e}")
