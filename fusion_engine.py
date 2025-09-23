"""
Fusion engine for scoring candidate ReID matches.

Provides:
 - L2 normalization helpers
 - spatial_score and time_decay
 - combined_score as: (alpha * emb_sim + beta * spatial_score) * time_decay
 - helper to pick best candidate from a list
"""

import numpy as np
import math
from typing import Optional

DEFAULT_ALPHA = 0.9
DEFAULT_BETA = 0.1
DEFAULT_THRESHOLD = 0.4521352216280379
DEFAULT_TAU = 30.0   # seconds for time decay
DEFAULT_VMAX = 3.0   # m/s
DEFAULT_PIXEL_SCALE = 500.0  # fallback pixel scale for spatial scoring if only pixel coords available

def l2_normalize(v):
    v = np.asarray(v, dtype=np.float32)
    n = np.linalg.norm(v)
    if n < 1e-12:
        return v
    
    return v / n

def time_decay(dt_seconds: float, tau: float = DEFAULT_TAU):
    if dt_seconds is None:
        return 1.0
    
    try:
        dt = float(dt_seconds)
    except Exception as e:
        dt = 0.0

    return math.exp(-dt / tau)

def spatial_score(candidate_world: Optional[tuple], query_world: Optional[tuple],
                  dt_seconds: float, vmax_m_s: float = DEFAULT_VMAX, pixel_mode=False,
                  pixel_scale=DEFAULT_PIXEL_SCALE):
    """
    Compute a spatial proximity score in [0,1].
    - If world coords provided (meters), use vmax_m_s * dt gating.
    - If not, but pixel coords are provided and pixel_mode True, use pixel_scale.
    """
    if candidate_world is None or query_world is None:
        return 0.0
    
    try:
        dx = float(candidate_world[0]) - float(query_world[0])
        dy = float(candidate_world[1]) - float(query_world[1])
    except Exception as e:
        return 0.0
    
    dist = math.hypot(dx, dy)
    if pixel_mode:
        # use pixel_scale as normalizer (tune per-cam)
        s = max(0.0, 1.0 - (dist / (pixel_scale * 3.0)))
        return s
    
    # world mode
    dt = max(0.001, dt_seconds if dt_seconds is not None else 1.0)
    max_allowed = max(1.0, vmax_m_s * dt)
    s = max(0.0, 1.0 - (dist / (max_allowed * 3.0)))

    return s

class FusionEngine:
    def __init__(self, alpha=DEFAULT_ALPHA, beta=DEFAULT_BETA,
                 threshold=DEFAULT_THRESHOLD, tau=DEFAULT_TAU,
                 vmax=DEFAULT_VMAX, pixel_mode=False, pixel_scale=DEFAULT_PIXEL_SCALE):
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.threshold = float(threshold)
        self.tau = float(tau)
        self.vmax = float(vmax)
        self.pixel_mode = bool(pixel_mode)
        self.pixel_scale = float(pixel_scale)

    def combined_score(self, emb_sim, cand_world, query_world, dt_seconds):
        """
        emb_sim expected to be cosine-like similarity in [-1,1] (we expect normalized embeddings).
        cand_world/query_world: (x,y) in meters or pixels depending on pixel_mode.
        dt_seconds: seconds difference between query ts and candidate ts (>=0)
        """
        if emb_sim is None:
            emb_sim = 0.0

        s_emb = float(emb_sim)
        s_spatial = spatial_score(cand_world, query_world, dt_seconds, self.vmax,
                                  pixel_mode=self.pixel_mode,
                                  pixel_scale=self.pixel_scale)
        base = (self.alpha * s_emb) + (self.beta * s_spatial)
        dec = time_decay(dt_seconds, tau=self.tau)

        return float(base * dec)

    def score_candidates(self, query_meta, candidates):
        """
        Score a list of candidates for one query.
        query_meta: dict with keys 'ts', 'world', 'camera_id', 'local_id', 'emb' (emb may not be used here)
        candidates: list of dicts each containing at least:
           { 'global_id', 'camera_id', 'ts', 'world_x', 'world_y', 'sim' }
        Returns: list of tuples (candidate, combined_score) sorted descending by score
        """
        q_ts = query_meta.get("ts", None)
        q_world = query_meta.get("world", None)
        scored = []
        for c in candidates:
            cand_ts = c.get("ts", None)
            # try to compute dt: query_ts - cand_ts
            dt = None
            try:
                if q_ts is not None and cand_ts is not None:
                    dt = float(q_ts) - float(cand_ts)
                    # ensure non-negative (if timestamps reversed, use abs)
                    if dt < 0:
                        dt = abs(dt)
                else:
                    dt = None
            except Exception as e:
                dt = None

            cand_world = None
            if c.get("world_x") is not None and c.get("world_y") is not None:
                cand_world = (c.get("world_x"), c.get("world_y"))

            score = self.combined_score(c.get("sim", 0.0), cand_world, q_world, dt)
            scored.append((c, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored

    def decide_best(self, query_meta, candidates):
        """
        Return best candidate (candidate_dict, score) if score >= threshold else (None, 0.0)
        """
        scored = self.score_candidates(query_meta, candidates)
        if not scored:
            return None, 0.0
        
        best = scored[0]
        if best[1] >= self.threshold:
            return best
        
        return None, best[1]