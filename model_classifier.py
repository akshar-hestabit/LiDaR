import os
import sys
import json
import numpy as np
import open3d as o3d
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any, List, Tuple

# ----------------------------
# PointNet++ Architecture Implementation
# ----------------------------

def square_distance(src, dst):
    """Calculate Euclidean distance between each two points."""
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist

def index_points(points, idx):
    """Index points according to the indices."""
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points

def farthest_point_sample(xyz, npoint):
    """Farthest point sampling."""
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    
    return centroids

def query_ball_point(radius, nsample, xyz, new_xyz):
    """Query ball point."""
    device = xyz.device
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape
    group_idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, S, 1])
    sqrdists = square_distance(new_xyz, xyz)
    group_idx[sqrdists > radius ** 2] = N
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]
    group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])
    mask = group_idx == N
    group_idx[mask] = group_first[mask]
    return group_idx

class PointNetSetAbstraction(nn.Module):
    def __init__(self, npoint, radius, nsample, in_channel, mlp, group_all):
        super(PointNetSetAbstraction, self).__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel
        self.group_all = group_all

    def forward(self, xyz, points):
        xyz = xyz.permute(0, 2, 1)
        if points is not None:
            points = points.permute(0, 2, 1)

        if self.group_all:
            new_xyz, new_points = self.sample_and_group_all(xyz, points)
        else:
            new_xyz, new_points = self.sample_and_group(xyz, points)
        
        new_points = new_points.permute(0, 3, 2, 1)  # [B, C+D, nsample,npoint]
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points = F.relu(bn(conv(new_points)))

        new_points = torch.max(new_points, 2)[0]
        new_xyz = new_xyz.permute(0, 2, 1)
        return new_xyz, new_points

    def sample_and_group(self, xyz, points):
        B, N, C = xyz.shape
        S = self.npoint
        fps_idx = farthest_point_sample(xyz, S)  # [B, npoint, C]
        new_xyz = index_points(xyz, fps_idx)
        idx = query_ball_point(self.radius, self.nsample, xyz, new_xyz)
        grouped_xyz = index_points(xyz, idx)  # [B, npoint, nsample, C]
        grouped_xyz_norm = grouped_xyz - new_xyz.view(B, S, 1, C)

        if points is not None:
            grouped_points = index_points(points, idx)
            new_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1)  # [B, npoint, nsample, C+D]
        else:
            new_points = grouped_xyz_norm
        return new_xyz, new_points

    def sample_and_group_all(self, xyz, points):
        device = xyz.device
        B, N, C = xyz.shape
        new_xyz = torch.zeros(B, 1, C).to(device)
        grouped_xyz = xyz.view(B, 1, N, C)
        if points is not None:
            new_points = torch.cat([grouped_xyz, points.view(B, 1, N, -1)], dim=-1)
        else:
            new_points = grouped_xyz
        return new_xyz, new_points

class PointNetFeaturePropagation(nn.Module):
    def __init__(self, in_channel, mlp):
        super(PointNetFeaturePropagation, self).__init__()
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv1d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm1d(out_channel))
            last_channel = out_channel

    def forward(self, xyz1, xyz2, points1, points2):
        xyz1 = xyz1.permute(0, 2, 1)
        xyz2 = xyz2.permute(0, 2, 1)

        points2 = points2.permute(0, 2, 1)
        B, N, C = xyz1.shape
        _, S, _ = xyz2.shape

        if S == 1:
            interpolated_points = points2.repeat(1, N, 1)
        else:
            dists = square_distance(xyz1, xyz2)
            dists, idx = dists.sort(dim=-1)
            dists, idx = dists[:, :, :3], idx[:, :, :3]  # [B, N, 3]

            dist_recip = 1.0 / (dists + 1e-8)
            norm = torch.sum(dist_recip, dim=2, keepdim=True)
            weight = dist_recip / norm
            interpolated_points = torch.sum(index_points(points2, idx) * weight.view(B, N, 3, 1), dim=2)

        if points1 is not None:
            points1 = points1.permute(0, 2, 1)
            new_points = torch.cat([points1, interpolated_points], dim=-1)
        else:
            new_points = interpolated_points

        new_points = new_points.permute(0, 2, 1)
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points = F.relu(bn(conv(new_points)))
        return new_points

# Batch Normalization with optional batch norm
class BN(nn.Module):
    def __init__(self, num_features, use_bn=True):
        super(BN, self).__init__()
        if use_bn:
            self.bn = nn.BatchNorm1d(num_features)
        else:
            self.bn = nn.Identity()
        
    def forward(self, x):
        return self.bn(x)

class FC(nn.Module):
    def __init__(self, in_features, out_features, use_bn=True, activation=True):
        super(FC, self).__init__()
        self.fc = nn.Linear(in_features, out_features)
        self.bn = BN(out_features, use_bn) if use_bn else nn.Identity()
        self.activation = activation
        
    def forward(self, x):
        x = self.fc(x)
        if hasattr(self, 'bn'):
            x = self.bn(x)
        if self.activation:
            x = F.relu(x)
        return x

class PointNet2Classification(nn.Module):
    def __init__(self, num_classes=40, normal_channel=False):
        super(PointNet2Classification, self).__init__()
        in_channel = 6 if normal_channel else 3
        self.normal_channel = normal_channel
        
        self.SA_modules = nn.ModuleList([
            PointNetSetAbstraction(npoint=512, radius=0.2, nsample=32, in_channel=in_channel, mlp=[32, 32, 64], group_all=False),
            PointNetSetAbstraction(npoint=128, radius=0.4, nsample=64, in_channel=64 + 3, mlp=[64, 64, 128], group_all=False),
            PointNetSetAbstraction(npoint=None, radius=None, nsample=None, in_channel=128 + 3, mlp=[128, 256, 1024], group_all=True)
        ])
        
        self.FC_layer = nn.ModuleList([
            FC(1024, 512, use_bn=True),
            nn.Dropout(0.5),
            FC(512, 256, use_bn=True),
            nn.Dropout(0.5),
            FC(256, num_classes, use_bn=False, activation=False)
        ])

    def forward(self, xyz):
        B, _, _ = xyz.shape
        if self.normal_channel:
            norm = xyz[:, 3:, :]
            xyz = xyz[:, :3, :]
        else:
            norm = None
        
        l1_xyz, l1_points = self.SA_modules[0](xyz, norm)
        l2_xyz, l2_points = self.SA_modules[1](l1_xyz, l1_points)
        l3_xyz, l3_points = self.SA_modules[2](l2_xyz, l2_points)
        
        x = l3_points.view(B, 1024)
        for layer in self.FC_layer:
            x = layer(x)
        
        return x

# Legacy PointNet for comparison
class PointNet(nn.Module):
    def __init__(self, num_classes):
        super(PointNet, self).__init__()
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)
        self.dropout = nn.Dropout(p=0.3)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=False)[0]
        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.dropout(self.fc2(x))))
        x = self.fc3(x)
        return x

# ----------------------------
# Checkpoint Analysis and Model Loading
# ----------------------------
def analyze_checkpoint(weights_path: str) -> Dict[str, Any]:
    """Analyze checkpoint structure to determine model architecture."""
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Weights not found at {weights_path}")
    
    print(f"📋 Analyzing checkpoint: {weights_path}")
    ckpt = torch.load(weights_path, map_location="cpu")
    
    # Print checkpoint structure
    print("🔍 Checkpoint keys:")
    for key in ckpt.keys():
        print(f"  - {key}: {type(ckpt[key])}")
    
    # Extract state dict
    if 'model_state' in ckpt:
        state_dict = ckpt['model_state']
        print("✅ Found 'model_state' key")
    elif 'state_dict' in ckpt:
        state_dict = ckpt['state_dict']
        print("✅ Found 'state_dict' key")
    elif 'model_state_dict' in ckpt:
        state_dict = ckpt['model_state_dict']
        print("✅ Found 'model_state_dict' key")
    elif 'model' in ckpt:
        state_dict = ckpt['model']
        print("✅ Found 'model' key")
    else:
        # Assume the entire checkpoint is the state dict
        state_dict = ckpt
        print("⚠️  Using entire checkpoint as state_dict")
    
    # Clean up keys (remove module. and model. prefixes)
    cleaned_state = {}
    for k, v in state_dict.items():
        new_key = k.replace('module.', '').replace('model.', '')
        cleaned_state[new_key] = v
    
    print(f"\n🧹 Cleaned state dict keys ({len(cleaned_state)} total):")
    fc_layers = []
    sa_modules = []
    for i, key in enumerate(sorted(cleaned_state.keys())):
        if hasattr(cleaned_state[key], 'shape'):
            print(f"  - {key}: {cleaned_state[key].shape}")
            if 'FC_layer' in key and 'weight' in key:
                fc_layers.append((key, cleaned_state[key].shape))
            if 'SA_modules' in key:
                sa_modules.append(key)
        else:
            print(f"  - {key}: {type(cleaned_state[key])}")
        
        if i >= 30:  # Limit output
            print(f"  ... and {len(cleaned_state)-30} more keys")
            break
    
    print(f"\n🔍 Found FC layers: {fc_layers}")
    print(f"🔍 Found SA modules: {len([k for k in sa_modules if 'SA_modules' in k])} layers")
    
    # Determine number of classes from the final FC layer
    num_classes = None
    final_fc_patterns = ['FC_layer.4.fc.weight', 'FC_layer.6.fc.weight', 'classifier.weight']
    
    for pattern in final_fc_patterns:
        if pattern in cleaned_state:
            num_classes = cleaned_state[pattern].shape[0]
            print(f"✅ Found final layer '{pattern}' with {num_classes} classes")
            break
    
    if num_classes is None:
        # Look for the last FC layer
        max_fc_idx = -1
        for key, shape in fc_layers:
            if 'FC_layer' in key:
                try:
                    idx = int(key.split('.')[1])
                    if idx > max_fc_idx:
                        max_fc_idx = idx
                        num_classes = shape[0]
                except:
                    continue
        if num_classes:
            print(f"🔍 Inferring {num_classes} classes from highest FC layer")
    
    # Check config for number of classes
    if num_classes is None and 'cfg' in ckpt:
        cfg = ckpt['cfg']
        if isinstance(cfg, dict):
            for key in ['num_classes', 'n_classes', 'classes', 'num_cls']:
                if key in cfg:
                    num_classes = cfg[key]
                    print(f"✅ Found {num_classes} classes in config['{key}']")
                    break
    
    if num_classes is None:
        print("⚠️  Could not determine number of classes, defaulting to 40")
        num_classes = 40
    
    # Check architecture type
    is_pointnet_plus = any('SA_modules' in key for key in cleaned_state.keys())
    has_normal_channel = any('6' in str(v.shape) for k, v in cleaned_state.items() if 'layer0.conv.weight' in k)
    
    analysis = {
        'state_dict': cleaned_state,
        'num_classes': num_classes,
        'is_pointnet_plus': is_pointnet_plus,
        'has_normal_channel': has_normal_channel,
        'original_keys': list(state_dict.keys()),
        'cleaned_keys': list(cleaned_state.keys())
    }
    
    return analysis

def load_model_from_checkpoint(weights_path: str) -> nn.Module:
    """Load model with automatic architecture detection."""
    analysis = analyze_checkpoint(weights_path)
    state_dict = analysis['state_dict']
    num_classes = analysis['num_classes']
    is_pointnet_plus = analysis['is_pointnet_plus']
    has_normal_channel = analysis['has_normal_channel']
    
    print(f"\n🏗️  Architecture: {'PointNet++' if is_pointnet_plus else 'PointNet'}")
    print(f"🏗️  Classes: {num_classes}")
    print(f"🏗️  Normal channels: {has_normal_channel}")
    
    # Try different model architectures
    models_to_try = []
    
    if is_pointnet_plus:
        models_to_try.extend([
            ('PointNet2Classification', lambda: PointNet2Classification(num_classes=num_classes, normal_channel=has_normal_channel)),
            ('PointNet2Classification_no_normal', lambda: PointNet2Classification(num_classes=num_classes, normal_channel=False)),
        ])
    
    models_to_try.append(('PointNet', lambda: PointNet(num_classes=num_classes)))
    
    for model_name, model_fn in models_to_try:
        try:
            print(f"🔄 Trying {model_name}...")
            model = model_fn()
            
            # Try to load state dict
            missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
            
            print(f"  Missing keys: {len(missing_keys)}")
            print(f"  Unexpected keys: {len(unexpected_keys)}")
            
            if len(missing_keys) == 0:
                print(f"✅ Successfully loaded {model_name} with no missing keys")
                model.eval()
                return model
            elif len(missing_keys) < 10:  # Allow some missing keys
                print(f"⚠️  Loaded {model_name} with {len(missing_keys)} missing keys")
                if len(missing_keys) <= 3:
                    print(f"    Missing: {missing_keys}")
                model.eval()
                return model
            else:
                print(f"❌ Too many missing keys ({len(missing_keys)}) for {model_name}")
                
        except Exception as e:
            print(f"❌ Failed to load {model_name}: {str(e)}")
    
    raise RuntimeError("Could not load model with any architecture")

def classify_object(cluster: o3d.geometry.PointCloud, model: nn.Module) -> int:
    """Classify a point cloud cluster."""
    pts = np.asarray(cluster.points, dtype=np.float32)
    N = pts.shape[0]
    
    # Handle point cloud size
    target_points = 1024
    if N < target_points:
        # Pad with duplicated points if too few
        if N > 0:
            pad_indices = np.random.choice(N, target_points - N, replace=True)
            pad_pts = pts[pad_indices]
            pts = np.vstack([pts, pad_pts])
        else:
            pts = np.zeros((target_points, 3), dtype=np.float32)
    else:
        # Sample if too many
        idx = np.random.choice(N, target_points, replace=False)
        pts = pts[idx]
    
    # Normalize
    pts = pts - pts.mean(axis=0)
    scale = np.max(np.linalg.norm(pts, axis=1))
    if scale > 0:
        pts = pts / scale
    
    # Classify
    with torch.no_grad():
        x = torch.from_numpy(pts).unsqueeze(0).transpose(2, 1)
        
        try:
            output = model(x)
            if isinstance(output, tuple):
                logits = output[0]
            else:
                logits = output
                
            if logits.dim() > 1:
                pred = logits.argmax(dim=1).item()
            else:
                pred = int(logits.item())
        except Exception as e:
            print(f"⚠️  Classification failed: {e}")
            pred = 0
            
    return pred

# ----------------------------
# Main Processing Pipeline
# ----------------------------
def process_point_cloud(pcd_path: Optional[str] = None,
                        weights_path: str = "/home/aksharrastogi/Downloads/model_best_test.pth",
                        output_dir: str = "output") -> str:
    """Process point cloud with object detection and classification."""
    
    # Load model
    try:
        model = load_model_from_checkpoint(weights_path)
        print(f"✅ Model loaded successfully")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return ""
    
    # Determine number of classes for labels
    if hasattr(model, 'FC_layer') and hasattr(model.FC_layer[-1], 'fc'):
        num_classes = model.FC_layer[-1].fc.out_features
    elif hasattr(model, 'fc3'):
        num_classes = model.fc3.out_features
    else:
        num_classes = 40  # Default
    
    class_labels = [f"class_{i}" for i in range(num_classes)]
    
    # Load point cloud
    if pcd_path and os.path.exists(pcd_path):
        print(f"📂 Loading point cloud from {pcd_path}")
        pcd = o3d.io.read_point_cloud(pcd_path)
    else:
        print("📂 Using sample point cloud")
        try:
            pcd = o3d.io.read_point_cloud(o3d.data.PCDPointCloud().path)
        except:
            # Create a simple sample if demo data fails
            points = np.random.rand(1000, 3)
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
    
    print(f"📊 Original point cloud: {len(pcd.points)} points")
    
    if len(pcd.points) == 0:
        print("❌ Empty point cloud")
        return ""
    
    # Preprocess
    pcd = pcd.voxel_down_sample(voxel_size=0.02)
    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    print(f"📊 After preprocessing: {len(pcd.points)} points")
    
    # Ground plane removal
    try:
        plane_model, inliers = pcd.segment_plane(distance_threshold=0.015,
                                                  ransac_n=3,
                                                  num_iterations=1000)
        objects = pcd.select_by_index(inliers, invert=True)
        print(f"📊 After ground removal: {len(objects.points)} points")
    except:
        print("⚠️  Ground plane removal failed, using all points")
        objects = pcd
    
    # Clustering
    if len(objects.points) > 0:
        labels = np.array(objects.cluster_dbscan(eps=0.03, min_points=10))
        max_label = int(labels.max()) if labels.size > 0 and labels.max() >= 0 else -1
        print(f"🔍 Detected {max_label+1} clusters")
    else:
        labels = np.array([])
        max_label = -1
        print("⚠️  No points for clustering")
    
    # Process clusters
    summary = []
    for lbl in range(max_label + 1):
        idxs = np.where(labels == lbl)[0]
        if len(idxs) < 10:  # Skip very small clusters
            continue
            
        cluster = objects.select_by_index(idxs)
        
        try:
            aabb = cluster.get_axis_aligned_bounding_box()
            center = aabb.get_center()
            extent = aabb.get_extent()
            
            # Classify cluster
            cid = classify_object(cluster, model)
            
            summary.append({
                "id": lbl,
                "position": [float(center[0]), float(center[1]), float(center[2])],
                "dimensions": [float(extent[0]), float(extent[1]), float(extent[2])],
                "class_id": int(cid),
                "class": class_labels[cid] if cid < len(class_labels) else f"class_{cid}",
                "point_count": len(idxs)
            })
            
            print(f"  Cluster {lbl}: {len(idxs)} points → {class_labels[cid] if cid < len(class_labels) else f'class_{cid}'}")
            
        except Exception as e:
            print(f"⚠️  Failed to process cluster {lbl}: {e}")
    
    # Save results
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "objects_summary.json")
    
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=4)
    
    print(f"💾 Results saved to {out_path}")
    print(f"📈 Summary: Found {len(summary)} valid objects")
    
    return out_path

if __name__ == "__main__":
    pc_path = sys.argv[1] if len(sys.argv) > 1 else None
    weights_path = sys.argv[2] if len(sys.argv) > 2 else "/home/aksharrastogi/Downloads/model_best_test.pth"
    
    result = process_point_cloud(pcd_path=pc_path, weights_path=weights_path)
    if result:
        print(f"🎉 Processing completed successfully!")
    else:
        print(f"❌ Processing failed!")