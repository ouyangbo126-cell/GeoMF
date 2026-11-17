"""GeoShip dataset and RegionPE geographic encoding."""

import os
import json
from PIL import Image
import math
import torch
from torch.utils.data import Dataset
import ast  # Safely parse the string as a list.


class GeoShipDataset(Dataset):
    """GeoShip images, numeric semantic attributes, and RegionPE features."""

    def __init__(self, root_dir, split='train', transform=None, args=None):
        self.args = args
        self.split = split
        self.transform = transform

        self.img_dir = os.path.join(root_dir, split)
        self.json_dir = os.path.join(root_dir, 'info')
        self.text_jsonl = os.path.join(root_dir, f"last/{split}.jsonl")

        self.feature_dict = self._load_feature_dict(self.text_jsonl)

        self.class_to_idx = {}
        self.samples = []
        self._load_samples()



    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, filename, label, metadata = self.samples[idx]

        # Load and transform the image.
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)

        key = os.path.splitext(os.path.basename(filename))[0]
        text = self.feature_dict.get(key, torch.zeros(19))  # Use a 19-element default vector.
        return img, text, label, metadata


    # ------------- Helper functions -------------
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
                    #print("before feature shape:", feature.shape)
                    #print("feature:", feature)
                    feature[-6:] = 0  # Zero out the trailing top-three entries.
                    #feature[11] = 0  # surrounding_facilities
                    #feature[12] = 0  # port
                    #print("after feature shape:", feature.shape)
                    #print("feature:", feature)
                    feature_dict[name] = feature
                except Exception as e:
                    print(f"[Error] Failed to parse feature for {name}: {e}")
        return feature_dict

    def _load_samples(self):
        class_dirs = sorted(os.listdir(self.img_dir))
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(class_dirs)}
        '''port_dict = {
            "Busan Naval Base": (129.0433179, 35.1035725),
            "Changi": (104.0256028, 1.3202668),
            "Alexander": (29.8581576, 31.1901641),
            "Dalian Port": (121.6612366, 38.9383918),
            "Everett": (-122.2286151, 47.984855),
            "Hebi": (135.3730436, 35.4769478),
            "Jinhae Naval Base": (128.6389822, 35.1497727),
            "Kaamba": (74.1420705, 14.7726255),
            "Lexington": (-97.3894329, 27.816035),
            "Mumbai": (72.8503197, 18.9601194),
            "Murmansk": (33.0644539, 68.9799089),
            "Norfolk Dockyard 1": (-76.2975999, 36.8161942),
            "North Island Naval Air Station": (-117.1889449, 32.7124003),
            "Pearl Harbor 1": (-157.9520793, 21.3538283),
            "Philadelphia": (-75.0769647, 39.9816701),
            "Plymouth Naval Harbor": (-4.1887608, 50.387151),
            "Portsmouth": (-1.097232, 50.8091807),
            "Qingdao Port": (120.2572932, 35.9980611),
            "San Francisco": (-122.4425391, 37.8070301),
            "Sanya Xingang": (109.4903022, 18.2324735),
            "Sasebo Base": (129.7172609, 33.1639522),
            "Sattahip Port": (100.9143178, 12.6279829),
            "South Carolina": (-79.9422911, 32.8325373),
            "Taranto 1": (17.2122624, 40.4737909),
            "Tianjin Museum": (117.743651, 38.9834576),
            "Toulon": (5.9324234, 43.1165805),
            "Wu Zhen": (132.6054154, 34.2193785),
            "Yokosuka": (139.7205442, 35.2404437),
            "bremerton base": (-122.6200587, 47.563737),
            "cochin port": (76.265535, 9.959162),
            "mayport base": (-81.4108004, 30.3902302),
            "newport news shipyard": (-76.4393137, 36.9808101),
            "norfolk naval base": (-76.3317281, 36.9439594),
            "pearl harbor 2": (-157.9906082, 21.3804226),
            "rio de janeiro": (-43.1723951, -22.8939789),
            "san diego naval base": (-117.1612597, 32.6782089)
        }'''
        '''port_dict = {
            "Busan Naval Base": (126.4251, 34.0506),
            "Changi": (85.6396, 11.5280),
            "Alexander": (13.4636, 47.4928),
            "Dalian Port": (126.4251, 34.0506),
            "Everett": (-131.0834, 34.4972),
            "Hebei": (126.4251, 34.0506),
            "Jinhae Naval Base": (126.4251, 34.0506),
            "Kaamba": (85.6396, 11.5280),
            "Lexington": (-80.4126, 34.5373),
            "Mumbai": (85.6396, 11.5280),
            "Murmansk": (13.4636, 47.4928),
            "Norfolk Dockyard 1": (-80.4126, 34.5373),
            "North Island Naval Air Station": (-131.0834, 34.4972),
            "Pearl Harbor 1": (-131.0834, 34.4972),
            "Philadelphia": (-80.4126, 34.5373),
            "Plymouth Naval Harbor": (13.4636, 47.4928),
            "Portsmouth": (13.4636, 47.4928),
            "Qingdao Port": (126.4251, 34.0506),
            "San Francisco": (-131.0834, 34.4972),
            "Sanya Xingang": (126.4251, 34.0506),
            "Sasebo Base": (126.4251, 34.0506),
            "Sattahip Port": (85.6396, 11.5280),
            "South Carolina": (-80.4126, 34.5373),
            "Taranto 1": (13.4636, 47.4928),
            "Tianjin Museum": (126.4251, 34.0506),
            "Toulon": (13.4636, 47.4928),
            "Wu Zhen": (126.4251, 34.0506),
            "Yokosuka": (126.4251, 34.0506),
            "bremerton base": (-131.0834, 34.4972),
            "cochin port": (85.6396, 11.5280),
            "mayport base": (-80.4126, 34.5373),
            "newport news shipyard": (-80.4126, 34.5373),
            "norfolk naval base": (-80.4126, 34.5373),
            "pearl harbor 2": (-131.0834, 34.4972),
            "rio de janeiro": (-43.1724, -22.8940),
            "san diego naval base": (-131.0834, 34.4972)
        }'''
        port_dict = {
            # Region 0: US East Coast (7 ports)
            "US East Coast": (-80.4126, 34.5373),
            # Region 1: East Asia (10 ports)
            "East Asia": (126.4251, 34.0506),
            # Region 2: Europe & Mediterranean (6 ports)
            "Europe Mediterranean": (13.4636, 47.4928),
            # Region 3: US West Coast & Pacific (7 ports)
            "US West Pacific": (-131.0834, 34.4972),
            # Region 4: Indian Ocean & SE Asia (5 ports)
            "Indian SE Asia": (85.6396, 11.5280),
            # Region 5: South America (1 port)
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
                    print('not found {json_path}')
                    continue

                try:
                    with open(json_path, 'r') as f:
                        meta = json.load(f)

                    lon = float(meta.get('lon', 0))
                    lat = float(meta.get('lat', 0))
                    port_lon, port_lat = nearest_region(lat, lon)
                    loc = []
                    loc += [float(lat) if lat else 0.0,
                            float(lon) if lon else 0.0,
                            float(port_lat) if port_lat else 0.0,
                            float(port_lon) if port_lon else 0.0]
                    '''try:
                        y, m, d = map(int, date_str.split('/'))
                        m = max(1, min(m, 12))
                        d = max(1, min(d, 31))
                        date = datetime.datetime(y, m, d)
                        loc += [self.get_scaled_date_ratio(date)]
                    except Exception as e:
                        print(f"[Warning] Invalid date format: {json_path}, {e}")
                        loc += [0.0]

                    while len(loc) < 3:
                        loc.append(0.0)'''

                    metadata = torch.tensor(self.encode_regionpe(loc), dtype=torch.float32)
                    label = self.class_to_idx[cls_name]

                    self.samples.append((img_path, file, label, metadata))

                except Exception as e:
                    print(f"[Error] Failed to process {json_path}: {e}")

    def encode_regionpe(self, loc):
        """Encode absolute and anchor-relative geographic descriptors with RegionPE."""
        if len(loc) != 4:
            print("[encode_regionpe] Warning: loc length != 4")
            return torch.zeros(self.input_dim)

        try:
            lat, lon, port_lat, port_lon = [float(x) for x in loc]
        except ValueError as err:
            print(f"[encode_regionpe] Error: {err}")
            return torch.zeros(self.input_dim)

        coarse_tensor = torch.tensor([lat, lon], dtype=torch.float32)
        coarse_sin = torch.sin(torch.pi * coarse_tensor)
        coarse_cos = torch.cos(torch.pi * coarse_tensor)

        geo_feature = [coarse_sin, coarse_cos]

        # Multiscale fine encoding for latitude and longitude
        k = 0.1
        for _ in range(3):
            fine_lat = (lat % k) / k
            fine_lon = (lon % k) / k
            fine_tensor = torch.tensor([fine_lat, fine_lon], dtype=torch.float32)
            fine_sin = torch.sin(torch.pi * fine_tensor)
            fine_cos = torch.cos(torch.pi * fine_tensor)
            geo_feature.append(fine_sin)
            geo_feature.append(fine_cos)
            k *= 10 # Use a finer scale.

        # Difference between the ship and port coordinates
        lat_diff = lat - port_lat
        lon_diff = lon - port_lon
        omega = 0.01

        diff_feature = []
        for _ in range(1):  # Encode differences at multiple omega scales.
            lat_diff_norm = (lat_diff % omega) / omega
            lon_diff_norm = (lon_diff % omega) / omega
            diff_tensor = torch.tensor([lat_diff_norm, lon_diff_norm], dtype=torch.float32)
            diff_sin = torch.sin(torch.pi * diff_tensor)
            diff_cos = torch.cos(torch.pi * diff_tensor)
            diff_feature.append(diff_sin)
            diff_feature.append(diff_cos)
            omega *= omega

        geo_feature.extend(diff_feature)
        # Add the distance feature in kilometers.
        distance_km = self.haversine_distance(lat, lon, port_lat, port_lon)
        distance_km = torch.tensor(distance_km/100, dtype=torch.float32)
        distance_sin = torch.sin(torch.pi * distance_km)
        distance_cos = torch.cos(torch.pi * distance_km)
        distance_tensor = torch.tensor([distance_sin, distance_cos], dtype=torch.float32)
        geo_feature.append(distance_tensor)

        # Bearing direction vector
        bearing_deg = self.compute_bearing(port_lat, port_lon, lat, lon)
        bearing_rad = math.radians(bearing_deg)
        dir_vec = torch.tensor([math.sin(bearing_rad), math.cos(bearing_rad)], dtype=torch.float32)
        geo_feature.append(dir_vec)

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


RASCDTextGraphEN = GeoShipDataset
