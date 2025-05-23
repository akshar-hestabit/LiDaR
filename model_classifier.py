import os
import sys
import json
import numpy as np
import open3d as o3d
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

# ----------------------------
# Full PointNet Architecture with STN and Feature Transform
# ----------------------------
class STN3d(nn.Module):
    def __init__(self):
        super(STN3d, self).__init__()
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 9)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

    def forward(self, x):
        batchsize = x.size(0)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=False)[0]

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = torch.eye(3, device=x.device).view(1, 9).repeat(batchsize, 1)
        x = x + iden
        return x.view(-1, 3, 3)

class PointNetfeat(nn.Module):
    def __init__(self, global_feat=True):
        super(PointNetfeat, self).__init__()
        self.stn = STN3d()
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.global_feat = global_feat

    def forward(self, x):
        B, D, N = x.size()
        trans = self.stn(x)
        x = x.transpose(2, 1)
        x = torch.bmm(x, trans)
        x = x.transpose(2, 1)
        x = F.relu(self.bn1(self.conv1(x)))
        pointfeat = x
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)
        if self.global_feat:
            return x
        else:
            x = x.view(-1, 1024, 1).repeat(1, 1, N)
            return torch.cat([x, pointfeat], 1)

class PointNet(nn.Module):
    def __init__(self, num_classes):
        super(PointNet, self).__init__()
        self.feat = PointNetfeat(global_feat=True)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.dropout = nn.Dropout(p=0.3)

    def forward(self, x):
        x = self.feat(x)
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.dropout(self.fc2(x))))
        x = self.fc3(x)
        return x

# ----------------------------
# Model Loading and Classification
# ----------------------------
def load_pretrained_pointnet(weights_path: str) -> PointNet:
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Weights not found at {weights_path}")

    ckpt = torch.load(weights_path, map_location="cpu")
    raw_state = ckpt.get('state_dict', ckpt)
    state_dict = {k.replace('module.', '').replace('model.', ''): v for k, v in raw_state.items()}

    fc3_key = 'fc3.weight'
    if fc3_key not in state_dict:
        raise KeyError(f"Cannot find '{fc3_key}' in checkpoint state_dict")
    num_classes = state_dict[fc3_key].shape[0]

    model = PointNet(num_classes=num_classes)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    print(f"✅ Loaded checkpoint with {num_classes} output classes.")
    return model

def classify_object(cluster: o3d.geometry.PointCloud, model: PointNet) -> int:
    pts = np.asarray(cluster.points, dtype=np.float32)
    N = pts.shape[0]
    if N < 1024:
        pad = np.zeros((1024 - N, 3), dtype=np.float32)
        pts = np.vstack([pts, pad])
    else:
        idx = np.random.choice(N, 1024, replace=False)
        pts = pts[idx]

    pts -= pts.mean(axis=0)
    pts /= np.max(np.linalg.norm(pts, axis=1))

    with torch.no_grad():
        x = torch.from_numpy(pts).unsqueeze(0).transpose(2, 1)
        out = model(x)
        pred = out.argmax(dim=1).item()
    return pred

# ----------------------------
# Main Processing Pipeline
# ----------------------------
def process_point_cloud(pcd_path: Optional[str] = None,
                        weights_path: str = "/home/aksharrastogi/Downloads/model_best_test.pth",
                        output_dir: str = "output") -> str:
    model = load_pretrained_pointnet(weights_path)
    class_labels = [f"class_{i}" for i in range(model.fc3.out_features)]

    if pcd_path and os.path.exists(pcd_path):
        pcd = o3d.io.read_point_cloud(pcd_path)
    else:
        print("Using sample point cloud")
        pcd = o3d.io.read_point_cloud(o3d.data.PCDPointCloud().path)

    pcd = pcd.voxel_down_sample(voxel_size=0.02)
    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)

    _, inliers = pcd.segment_plane(distance_threshold=0.015,
                                   ransac_n=3,
                                   num_iterations=1000)
    objects = pcd.select_by_index(inliers, invert=True)

    labels = np.array(objects.cluster_dbscan(eps=0.03, min_points=10))
    max_label = int(labels.max()) if labels.size > 0 else -1
    print(f"Detected {max_label+1} clusters.")

    summary = []
    for lbl in range(max_label + 1):
        idxs = np.where(labels == lbl)[0]
        cluster = objects.select_by_index(idxs)
        aabb = cluster.get_axis_aligned_bounding_box()
        cid = classify_object(cluster, model)
        summary.append({
            "id": lbl,
            "position": list(aabb.get_center()),
            "dimensions": list(aabb.get_extent()),
            "class": class_labels[cid]
        })

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "objects_summary.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=4)
    print(f"Results saved to {out_path}")
    return out_path

if __name__ == "__main__":
    pc_path = sys.argv[1] if len(sys.argv) > 1 else None
    process_point_cloud(pcd_path=pc_path)
