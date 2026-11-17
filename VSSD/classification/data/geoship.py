"""GeoShip dataset and RegionPE geographic encoding."""

import os
import json
import numpy as np
from PIL import Image
import random
import math
import torch
from torch.utils.data import Dataset
import ast  # Safely parse the string as a list.


def meters_to_degrees_lat(meters, lat):
    """Convert meters to degrees of latitude."""
    return meters / 111320.0

def meters_to_degrees_lon(meters, lat):
    """Convert meters to degrees of longitude at the given latitude."""
    return meters / (111320.0 * math.cos(math.radians(lat)))


class GeoShipDataset(Dataset):
    """GeoShip inputs with support for the iNaturalist experiments."""

    def __init__(self, root_dir, split='train', transform=None, args=None, meta_dim=19):
        self.args = args
        self.split = split
        self.transform = transform
        # Coordinate noise standard deviation in meters. A value of zero disables noise.
        if args is not None and hasattr(args, 'DATA') and hasattr(args.DATA, 'SIGMA'):
            self.sigma = args.DATA.SIGMA
        else:
            self.sigma = 0.0
        print(f"[GeoShipDataset] split={split}, sigma={self.sigma}m")

        self.root_dir = root_dir
        self.img_dir = os.path.join(root_dir, split)
        
        # Detect the dataset type automatically.
        self.is_inat = False
        self.json_data = None
        
        # Check whether the dataset is iNaturalist.
        inat_json = os.path.join(root_dir, f"{split}_mini.json")
        if not os.path.exists(inat_json):
            inat_json = os.path.join(root_dir, f"val.json") if split == 'val' else None
        
        if inat_json and os.path.exists(inat_json):
            print(f"[GeoShipDataset] Loading JSON from {inat_json}...")
            with open(inat_json, 'r', encoding='utf-8') as f:
                self.json_data = json.load(f)
            print(f"[GeoShipDataset] JSON loaded, {len(self.json_data['images'])} images")
            if 'images' in self.json_data and 'categories' in self.json_data:
                if len(self.json_data['images']) > 0 and 'latitude' in self.json_data['images'][0]:
                    self.is_inat = True
                    print(f"[GeoShipDataset] Detected iNaturalist dataset")
        
        if self.is_inat:
            self._init_inat()
        else:
            self._init_rascd()

    def _init_inat(self):
        """Initialize the iNaturalist dataset."""
        self.json_dir = None
        self.text_jsonl = None
        self.feature_dict = {}
        
        # Map image IDs to metadata.
        self.id_to_info = {img['id']: img for img in self.json_data['images']}
        
        print(f"[GeoShipDataset] Scanning train directory...")
        # Read class names from the train directory in ImageFolder alphabetical order.
        class_dirs = sorted(os.listdir(self.img_dir))
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(class_dirs)}
        class_set = set(class_dirs)  # Store entries for fast lookup.
        
        print(f"[GeoShipDataset] Building image index ({len(self.json_data['images'])} images)...")
        # Map image IDs to labels using the class directory from each path.
        self.filename_to_label = {}
        for img_info in self.json_data['images']:
            file_name = img_info['file_name']
            # file_name format: train_mini/00000_Animalia_.../abc.jpg
            # Alternate format: train/00000_Animalia_.../abc.jpg
            parts = file_name.replace('train_mini/', 'train/').split('/')
            if len(parts) >= 2:
                dir_name = parts[-2]  # Use the second-to-last path component as the directory name.
                if dir_name in class_set:
                    self.filename_to_label[img_info['id']] = self.class_to_idx[dir_name]
        
        print(f"[GeoShipDataset] Found {len(self.class_to_idx)} classes, {len(self.filename_to_label)} images matched")
        
        self.samples = []
        print(f"[GeoShipDataset] Loading samples...")
        self._load_samples()
        print(f"[GeoShipDataset] Loaded {len(self.samples)} samples")

    def _init_rascd(self):
        """Initialize the RASCD dataset."""
        self.json_dir = os.path.join(self.root_dir, 'info')
        self.text_jsonl = os.path.join(self.root_dir, f"last/{self.split}.jsonl")
        self.feature_dict = self._load_feature_dict(self.text_jsonl)
        
        self.class_to_idx = {}
        self.samples = []
        self._load_samples()

    def _shuffle_metadata(self, seed=42):
        """Shuffle metadata once for bias control (reproducible)"""
        random.seed(seed)
        metadata_list = [s[3] for s in self.samples]
        random.shuffle(metadata_list)
        self.samples = [
            (p, f, l, m) for (p, f, l, _), m in zip(self.samples, metadata_list)
        ]
        self._metadata_shuffle_seed = seed

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, filename, label, metadata = self.samples[idx]

        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)

        if self.is_inat:
            text = torch.ones(19)
        else:
            key = os.path.splitext(os.path.basename(filename))[0]
            text = self.feature_dict.get(key, torch.zeros(19))

        return img, text, label, metadata


    # ------------- RASCD helper functions -------------
    def _load_feature_dict(self, path):
        feature_dict = {}
        if not os.path.exists(path):
            print(f"[Warning] Missing feature jsonl: {path}")
            return feature_dict
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                item = json.loads(line.strip())
                name = os.path.splitext(os.path.basename(item["image_name"]))[0]
                try:
                    feature_list = ast.literal_eval(item["content"])
                    feature = torch.tensor(feature_list, dtype=torch.float)
                    feature[-6:] = 0
                    feature_dict[name] = feature
                except Exception as e:
                    print(f"[Error] Failed to parse feature for {name}: {e}")
        return feature_dict

    def _load_samples(self):
        if self.is_inat:
            self._load_samples_inat()
        else:
            self._load_samples_rascd()

    def _load_samples_inat(self):
        """Load iNaturalist samples without checking whether files exist."""
        # Read the configuration once before iterating.
        ebase, eoff, ediff, edist, edir, input_dim = self.regionpe_config()
        
        for img_id, info in self.id_to_info.items():
            if img_id not in self.filename_to_label:
                continue
            
            label = self.filename_to_label[img_id]
            file_name = info['file_name']
            
            # Map train_mini/xxx paths to train/xxx.
            file_name = file_name.replace('train_mini/', 'train/')
            img_path = os.path.join(self.root_dir, file_name)
            
            # Parse the geographic coordinates.
            lat = info.get('latitude')
            lon = info.get('longitude')
            
            if lat is None or lon is None or (lat == 0 and lon == 0):
                lat, lon, port_lat, port_lon = 0.0, 0.0, 0.0, 0.0
            else:
                port_lat, port_lon = lat, lon
            
            # Call _encode_regionpe directly to reuse the configuration.
            metadata = self._encode_regionpe(lat, lon, port_lat, port_lon, ebase, eoff, ediff, edist, edir, input_dim)
            self.samples.append((img_path, os.path.basename(file_name), label, metadata))
    
    def _encode_loc_inat(self, lat, lon, ebase, eoff, input_dim):
        """GeoShip inputs with support for the iNaturalist experiments."""

        geo_feature = []

        # Ebase: absolute coordinate encoding (2 coordinates x 2 = 4 dimensions)
        if ebase:
            coarse_tensor = torch.tensor([lat, lon], dtype=torch.float32)
            coarse_sin = torch.sin(torch.pi * coarse_tensor)
            coarse_cos = torch.cos(torch.pi * coarse_tensor)
            geo_feature.extend([coarse_sin, coarse_cos])

        # Eoff: multiscale relative offsets (3 scales x 2 coordinates x 2 = 12 dimensions)
        if eoff:
            k = 0.1
            for _ in range(3):
                fine_lat = (lat % k) / k
                fine_lon = (lon % k) / k
                fine_tensor = torch.tensor([fine_lat, fine_lon], dtype=torch.float32)
                fine_sin = torch.sin(torch.pi * fine_tensor)
                fine_cos = torch.cos(torch.pi * fine_tensor)
                geo_feature.extend([fine_sin, fine_cos])
                k *= 10

        if not geo_feature:
            return torch.zeros(input_dim)
        
        return torch.cat(geo_feature)

    def _load_samples_rascd(self):
        """Load RASCD samples."""
        class_dirs = sorted(os.listdir(self.img_dir))
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(class_dirs)}
        
        port_dict = {
            "US East Coast": (-80.4126, 34.5373),
            "East Asia": (126.4251, 34.0506),
            "Europe Mediterranean": (13.4636, 47.4928),
            "US West Pacific": (-131.0834, 34.4972),
            "Indian SE Asia": (85.6396, 11.5280),
            "South America": (-43.1724, -22.8940),
        }

        def haversine(lat1, lon1, lat2, lon2):
            R = 6371.0
            phi1, phi2 = math.radians(lat1), math.radians(lat2)
            dphi = math.radians(lat2 - lat1)
            dlambda = math.radians(lon2 - lon1)
            a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
            return R * 2 * math.asin(math.sqrt(a))

        def nearest_region(lat, lon):
            min_dist, nearest = float('inf'), None
            for name, (rlon, rlat) in port_dict.items():
                d = haversine(lat, lon, rlat, rlon)
                if d < min_dist:
                    min_dist, nearest = d, (rlon, rlat)
            return nearest

        for cls_name in class_dirs:
            cls_img_dir = os.path.join(self.img_dir, cls_name)
            cls_json_dir = os.path.join(self.json_dir, cls_name)
            if not os.path.isdir(cls_img_dir):
                continue

            for file in os.listdir(cls_img_dir):
                if not file.endswith('.jpg'):
                    continue
                if 'placeholder' in file.lower():
                    continue

                img_path = os.path.join(cls_img_dir, file)
                json_path = os.path.join(cls_json_dir, file.replace('.jpg', '.json'))
                if not os.path.exists(json_path):
                    continue

                try:
                    with open(json_path, 'r') as f:
                        meta = json.load(f)

                    lat = float(meta.get('lat', 0))
                    lon = float(meta.get('lon', 0))

                    if self.sigma > 0:
                        noise_lat_m = np.random.normal(0, self.sigma)
                        noise_lon_m = np.random.normal(0, self.sigma)
                        noise_lat_deg = meters_to_degrees_lat(noise_lat_m, lat)
                        noise_lon_deg = meters_to_degrees_lon(noise_lon_m, lat)
                        lat = np.clip(lat + noise_lat_deg, -90, 90)
                        lon = np.clip(lon + noise_lon_deg, -180, 180)

                    port_lon, port_lat = nearest_region(lat, lon)
                    loc = [float(lat) if lat else 0.0,
                           float(lon) if lon else 0.0,
                           float(port_lat) if port_lat else 0.0,
                           float(port_lon) if port_lon else 0.0]
                    metadata = torch.tensor(self.encode_regionpe(loc), dtype=torch.float32)
                    label = self.class_to_idx[cls_name]
                    self.samples.append((img_path, file, label, metadata))

                except Exception as e:
                    print(f"[Error] Failed to process {json_path}: {e}")

    def regionpe_config(self):
        """Return the five enabled RegionPE component flags and the input dimension."""
        ebase = getattr(self.args, 'DATA', None) and getattr(self.args.DATA, 'REGION_PE_EBASE', True)
        eoff = getattr(self.args, 'DATA', None) and getattr(self.args.DATA, 'REGION_PE_EOFF', True)
        ediff = getattr(self.args, 'DATA', None) and getattr(self.args.DATA, 'REGION_PE_EDIFF', True)
        edist = getattr(self.args, 'DATA', None) and getattr(self.args.DATA, 'REGION_PE_EDIST', True)
        edir = getattr(self.args, 'DATA', None) and getattr(self.args.DATA, 'REGION_PE_EDIR', True)
        
        # Compute input_dim from the dimensions of the enabled components.
        # Ebase: 2 coords * 2 (sin+cos) = 4 dims
        # Eoff: 3 scales * 2 coords * 2 (sin+cos) = 12 dims
        # Ediff: 1 layer * 2 coords * 2 (sin+cos) = 4 dims
        # Edist: 2 (sin+cos) = 2 dims
        # Edir: 2 (sin+cos) = 2 dims
        input_dim = 0
        if ebase:
            input_dim += 4
        if eoff:
            input_dim += 12
        if ediff:
            input_dim += 4
        if edist:
            input_dim += 2
        if edir:
            input_dim += 2
        
        return ebase, eoff, ediff, edist, edir, max(input_dim, 1)

    def encode_regionpe(self, loc):
        """Encode absolute and anchor-relative geographic descriptors with RegionPE."""
        ebase, eoff, ediff, edist, edir, input_dim = self.regionpe_config()
        
        if len(loc) != 4:
            print("[encode_regionpe] Warning: loc length != 4")
            return torch.zeros(input_dim)

        try:
            lat, lon, port_lat, port_lon = [float(x) for x in loc]
        except ValueError as err:
            print(f"[encode_regionpe] Error: {err}")
            return torch.zeros(input_dim)
        
        return self._encode_regionpe(lat, lon, port_lat, port_lon, ebase, eoff, ediff, edist, edir, input_dim)
    
    def _encode_regionpe(self, lat, lon, port_lat, port_lon, ebase, eoff, ediff, edist, edir, input_dim):
        """Encode geographic coordinates using the supplied configuration."""
        geo_feature = []

        # Ebase: absolute coordinate encoding (2 coordinates x 2 = 4 dimensions)
        if ebase:
            coarse_tensor = torch.tensor([lat, lon], dtype=torch.float32)
            coarse_sin = torch.sin(torch.pi * coarse_tensor)
            coarse_cos = torch.cos(torch.pi * coarse_tensor)
            geo_feature.extend([coarse_sin, coarse_cos])

        # Eoff: multiscale relative offsets (3 scales x 2 coordinates x 2 = 12 dimensions)
        if eoff:
            k = 0.1
            for _ in range(3):
                fine_lat = (lat % k) / k
                fine_lon = (lon % k) / k
                fine_tensor = torch.tensor([fine_lat, fine_lon], dtype=torch.float32)
                fine_sin = torch.sin(torch.pi * fine_tensor)
                fine_cos = torch.cos(torch.pi * fine_tensor)
                geo_feature.extend([fine_sin, fine_cos])
                k *= 10

        # Ediff: anchor-relative displacement (1 level x 2 coordinates x 2 = 4 dimensions)
        if ediff:
            lat_diff = lat - port_lat
            lon_diff = lon - port_lon
            omega = 0.01
            lat_diff_norm = (lat_diff % omega) / omega
            lon_diff_norm = (lon_diff % omega) / omega
            diff_tensor = torch.tensor([lat_diff_norm, lon_diff_norm], dtype=torch.float32)
            diff_sin = torch.sin(torch.pi * diff_tensor)
            diff_cos = torch.cos(torch.pi * diff_tensor)
            geo_feature.extend([diff_sin, diff_cos])

        # Edist: geodesic distance encoding (2 dimensions)
        if edist:
            distance_km = self.haversine_distance(lat, lon, port_lat, port_lon)
            distance_km = torch.tensor(distance_km/100, dtype=torch.float32)
            distance_sin = torch.sin(torch.pi * distance_km)
            distance_cos = torch.cos(torch.pi * distance_km)
            distance_tensor = torch.tensor([distance_sin, distance_cos], dtype=torch.float32)
            geo_feature.append(distance_tensor)

        # Edir: bearing direction encoding (2 dimensions)
        if edir:
            bearing_deg = self.compute_bearing(port_lat, port_lon, lat, lon)
            bearing_rad = math.radians(bearing_deg)
            dir_vec = torch.tensor([math.sin(bearing_rad), math.cos(bearing_rad)], dtype=torch.float32)
            geo_feature.append(dir_vec)

        if not geo_feature:
            return torch.zeros(input_dim)
        
        geo_feature = torch.cat(geo_feature)
        return geo_feature

    def compute_bearing(self, lat1, lon1, lat2, lon2):
        """
        Compute the bearing from (lat1, lon1) to (lat2, lon2).
        Return an angle in degrees in the range [0, 360).
        """
        lat1_rad = math.radians(lat1)
        lat2_rad = math.radians(lat2)
        dlon_rad = math.radians(lon2 - lon1)

        y = math.sin(dlon_rad) * math.cos(lat2_rad)
        x = math.cos(lat1_rad) * math.sin(lat2_rad) - \
            math.sin(lat1_rad) * math.cos(lat2_rad) * math.cos(dlon_rad)
    
        brng_rad = math.atan2(y, x)
        brng_deg = (math.degrees(brng_rad) + 360) % 360
        return brng_deg

    def haversine_distance(self, lat1, lon1, lat2, lon2):
        R = 6371.0  # Earth radius in kilometers
        phi1 = math.radians(lat1)
        phi2 = math.radians(lat2)
        d_phi = math.radians(lat2 - lat1)
        d_lambda = math.radians(lon2 - lon1)

        a = math.sin(d_phi / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(d_lambda / 2.0) ** 2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

        return R * c

    def _is_leap_year(self, year):
        return year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)

    def get_scaled_date_ratio(self, date_time):
        days = [31, 28 + self._is_leap_year(date_time.year), 31, 30, 31, 30,
                31, 31, 30, 31, 30, 31]
        total_days = sum(days)
        month, day = date_time.month, date_time.day
        sum_days = sum(days[:month - 1]) + day
        return (sum_days / total_days) * 2 - 1

    encode_loc_with_global_and_lpc = encode_regionpe
    _encode_loc_fast = _encode_regionpe
    _get_geo_config = regionpe_config


RASCDTextGraphEN = GeoShipDataset
