# from jemas_bytetracker.bytetracker import ByteTracker
from deep_sort_realtime.deepsort_tracker import DeepSort

class TrackerWrapper:
    def __init__(self, max_age=30, n_init=3):
        ## Can also use deepstream tracker
        ## Remove this if you do or just use it as a parser
        # self.tracker = ByteTracker()
        self.tracker = DeepSort(max_age=max_age, n_init=n_init, nms_max_overlap=1.0)

    def update(self, detections, frame):
        # detections: list of [x1,y1,x2,y2,score, class, embedding(optional)] 
        ds_dets = []
        for d in detections:
            x1,y1,x2,y2,s,cls, emb = d
            ds_dets.append(([x1,y1,x2,y2], s, cls, emb))

        tracks = self.tracker.update_tracks(ds_dets, frame=frame)
        out=[]
        for t in tracks:
            if not t.is_confirmed(): continue
            out.append({
                "track_id": t.track_id, "bbox": t.to_ltwh(),
                "last_obs": t.last_observation, "emb": t.last_embedding
            })

        return out