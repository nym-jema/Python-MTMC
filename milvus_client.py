"""
Simple Milvus helper for ReID embeddings.

- Creates collection (if missing) with fields:
  - emb: FLOAT_VECTOR (dim)
  - global_id: VARCHAR
  - camera_id: VARCHAR
  - ts: DOUBLE
  - world_x/world_y: DOUBLE

- Creates HNSW index (IP metric) with recommended defaults: M=16, efConstruction=200.
- Supports insert() and search().
"""

from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, utility
import numpy as np
import time

class MilvusClient:
    def __init__(self, host="mdx-milvus-db", port="19530", collection_name="mdx_reid_collection", dim=256,
                 index_params=None, create_if_missing=True):
        self.host = host
        self.port = str(port)
        self.collection_name = collection_name
        self.dim = dim
        self._col = None

        # default index params recommended from your analysis (HNSW)
        if index_params is None:
            index_params = {
                "index_type": "HNSW",
                "metric_type": "IP",   # use IP with L2-normalized vectors = cosine
                "params": {"M": 16, "efConstruction": 200}
            }
        self.index_params = index_params

        connections.connect("default", host=self.host, port=self.port)
        if utility.has_collection(self.collection_name):
            self._col = Collection(self.collection_name)
        else:
            if create_if_missing:
                self._create_collection()
            else:
                raise RuntimeError(f"Collection {self.collection_name} not found and create_if_missing=False")

    def _create_collection(self):
        fields = [
            FieldSchema(name="emb", dtype=DataType.FLOAT_VECTOR, dim=self.dim, description="embedding"),
            FieldSchema(name="global_id", dtype=DataType.VARCHAR, max_length=128),
            FieldSchema(name="camera_id", dtype=DataType.VARCHAR, max_length=128),
            FieldSchema(name="ts", dtype=DataType.DOUBLE),
            FieldSchema(name="world_x", dtype=DataType.DOUBLE),
            FieldSchema(name="world_y", dtype=DataType.DOUBLE)
        ]
        schema = CollectionSchema(fields, description="ReID embeddings collection")
        col = Collection(self.collection_name, schema)
        
        # create index on embedding
        col.create_index(field_name="emb", index_params=self.index_params)
        col.load()
        self._col = col
        time.sleep(0.5)

    def insert(self, vectors, metas, flush=True):
        """
        Insert vectors and metadata.

        vectors: NxD numpy array or list of lists (assumed already L2-normalized if using IP)
        metas: list of dicts with keys: global_id (str), camera_id (str), ts (float), world_x (float), world_y (float)
        """
        if self._col is None:
            raise RuntimeError("Collection not initialized")
        if isinstance(vectors, np.ndarray):
            vecs = vectors.tolist()
        else:
            vecs = vectors

        global_ids = [m.get("global_id", "") for m in metas]
        camera_ids = [m.get("camera_id", "") for m in metas]
        ts = [float(m.get("ts", 0.0) or 0.0) for m in metas]
        world_x = [float(m.get("world_x", 0.0) or 0.0) for m in metas]
        world_y = [float(m.get("world_y", 0.0) or 0.0) for m in metas]
        entities = [vecs, global_ids, camera_ids, ts, world_x, world_y]
        res = self._col.insert(entities)
        if flush:
            self._col.flush()

        return res

    def search(self, query_vectors, top_k=10, ef_search=200, output_fields=None, expr=None):
        """
        Search vectors.

        query_vectors: list of vectors or numpy array (Q x D)
        Returns search results (pymilvus style).
        """
        if output_fields is None:
            output_fields = ["global_id", "camera_id", "ts", "world_x", "world_y"]

        search_params = {"metric_type": self.index_params.get("metric_type", "IP"), "params": {"ef": ef_search}}
        results = self._col.search(
            query_vectors.tolist() if hasattr(query_vectors, "tolist") else query_vectors,
            "emb", param=search_params, limit=top_k, expr=expr, output_fields=output_fields
            )
        return results

    def create_index(self, index_params=None):
        if index_params is None:
            index_params = self.index_params

        self._col.create_index(field_name="emb", index_params=index_params)

    def drop_collection(self):
        utility.drop_collection(self.collection_name)

    def get_collection(self):
        return self._col





"""
## milvus_setup.py
from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, utility

MILVUS_HOST = "milvus-db"   # or host IP if host-mode
MILVUS_PORT = "19530"
COLLECTION_NAME = "reid_collection"
DIM = 256

connections.connect("default", host=MILVUS_HOST, port=MILVUS_PORT)

if not utility.has_collection(COLLECTION_NAME):
    fields = [
        FieldSchema(name="emb", dtype=DataType.FLOAT_VECTOR, dim=DIM),
        FieldSchema(name="global_id", dtype=DataType.VARCHAR, max_length=64),
        FieldSchema(name="camera_id", dtype=DataType.VARCHAR, max_length=64),
        FieldSchema(name="ts", dtype=DataType.DOUBLE),
        FieldSchema(name="world_x", dtype=DataType.DOUBLE),
        FieldSchema(name="world_y", dtype=DataType.DOUBLE),
    ]
    schema = CollectionSchema(fields, description="ReID embedding collection")
    col = Collection(COLLECTION_NAME, schema=schema)

    # create HNSW index with recommended params
    index_params = {
        "index_type": "HNSW",
        "metric_type": "IP",                     # use IP with L2-normalized vectors (approx cosine)
        "params": {"M": 16, "efConstruction": 200}
    }
    col.create_index(field_name="emb", index_params=index_params)
    col.load()
    print("Created collection and index:", COLLECTION_NAME)
else:
    col = Collection(COLLECTION_NAME)
    print("Collection exists:", COLLECTION_NAME)


def insert_vectors(col, vectors, metas):
    '''
    vectors: list of lists (N x DIM) already L2-normalized
    metas: list of dicts with keys: global_id (str), camera_id (str), ts (float), world_x (float), world_y (float)
    '''
    global_ids = [m["global_id"] for m in metas]
    camera_ids = [m["camera_id"] for m in metas]
    ts = [m["ts"] for m in metas]
    world_x = [m.get("world_x", 0.0) for m in metas]
    world_y = [m.get("world_y", 0.0) for m in metas]
    entities = [vectors, global_ids, camera_ids, ts, world_x, world_y]
    _ = col.insert(entities)
    # flush if you want durability
    col.flush()


## Query
search_params = {"metric_type": "IP", "params": {"ef": 200}}   # efSearch = 200 recommended
limit = 10
expr = None  # optional expression to filter candidates by ts or camera

results = col.search([query_vec], "emb", param=search_params, limit=limit, expr=expr,
                     output_fields=["global_id", "camera_id", "ts", "world_x", "world_y"])

# results is a list (per query). For a single query:
hits = results[0]
for h in hits:
    # depending on pymilvus version the hit object gives id/distance and entity field values
    sim = getattr(h, "score", None) or getattr(h, "distance", None)
    # prefer calling get() for output fields:
    metadata = {k: h.entity.get(k) if h.entity else None for k in ["global_id", "camera_id", "ts", "world_x", "world_y"]}
    # compute dt = query_ts - metadata['ts']
    # compute candidate_world = (metadata['world_x'], metadata['world_y'])
    # compute combined_score(sim, candidate_world, query_world, dt) as above
"""