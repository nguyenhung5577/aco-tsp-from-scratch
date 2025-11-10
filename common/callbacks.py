# common/callbacks.py
from typing import Optional, Dict, Any, List

def make_logger():
    log = {
        "iter": [],               # [0, 1, 2, ...]
        "best_len": [],           # [L_best(t)]
        "best_tour": [],          # [tour_best(t)]
        "extras": []              # vd: {"tau": tau.copy()} cho ACO
    }
    def cb(iter_idx: int, best_len: float, best_tour: List[int], extras: Optional[Dict[str, Any]]=None):
        log["iter"].append(iter_idx)
        log["best_len"].append(float(best_len))
        log["best_tour"].append(best_tour[:])
        log["extras"].append(extras or {})
    return log, cb
