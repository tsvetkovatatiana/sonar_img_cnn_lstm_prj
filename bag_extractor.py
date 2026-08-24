#!/usr/bin/env python3

# to run this file you need to be on ubuntu 20.04 version and follow instructions in readme


import os
import rosbag
import pandas as pd
import numpy as np
from cv_bridge import CvBridge
import cv2
import functools

SAMPLE_ONLY = False
DATA_DIR = "data"


def ensure_dir(directory):
    os.makedirs(directory, exist_ok=True)


def write_image(filepath, msg, bridge):
    if hasattr(msg, "format") and "compressed" in msg.format:
        buf = np.frombuffer(msg.data, dtype=np.uint8)
        cv_img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    else:
        cv_img = bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")

    if cv_img is None:
        return {}

    cv2.imwrite(filepath, cv_img)

    return {
        "height": cv_img.shape[0],
        "width": cv_img.shape[1],
    }


def open_bag(bag_path):
    try:
        return rosbag.Bag(bag_path)
    except Exception as e:
        print(f"Skipping: {bag_path} ({e})")
        return None


def process_bag(bag_path):
    print(f"\nProcessing: {bag_path}")

    bag = open_bag(bag_path)
    if bag is None:
        return

    output_dir = bag_path.replace(".bag", "_extract")
    ensure_dir(output_dir)

    bridge = CvBridge()

    types, topics = bag.get_type_and_topic_info()

    print("Message count:", bag.get_message_count())
    print("Topics:", list(topics.keys()))
    print("Output:", output_dir)

    SYNC_TOPIC = "/camera_crop/image_rect_color/compressed"

    config = {
        "/speed": {
            "values": ["cog", "sog"],
            "prefix": "speed_",
        },
        "/waterdepth": {
            "values": ["waterdepth"],
            "prefix": "waterdepth_",
        },
    }

    config["/camera_crop/image_rect_color/compressed"] = {
        "values": [],
        "type": "image",
        "prefix": "camera_",
    }

    for topic, conf in config.items():
        if conf.get("type") == "image":
            ensure_dir(os.path.join(output_dir, conf["prefix"] + "images"))

    rows = {topic: [] for topic in config.keys()}

    msgs = bag.read_messages(list(config.keys()))

    i = 0
    for topic, msg, t in msgs:
        if topic not in config:
            continue

        conf = config[topic]

        timestamp = t.to_nsec()

        row = {
            "timestamp": timestamp,
            "datetime": pd.to_datetime(timestamp),
        }

        if conf.get("type") == "image":
            filename = f"{topic.replace('/', '_')}_{msg.header.seq:09d}.jpg"
            rel_path = os.path.join(conf["prefix"] + "images", filename)
            full_path = os.path.join(output_dir, rel_path)

            img_info = write_image(full_path, msg, bridge)
            row.update(img_info)
            row["filepath"] = rel_path
            row["filename"] = filename

        for col in conf.get("values", []):
            row[col] = getattr(msg, col, np.nan)

        rows[topic].append(row)

        i += 1
        if i % 50 == 0:
            print(".", end="", flush=True)

        if SAMPLE_ONLY and i > 100:
            break

    bag.close()

    print("\nWriting CSV files...")

    dfs = {}
    for topic, data in rows.items():
        df = pd.DataFrame(data)

        if len(df) > 0:
            df = df.sort_values("timestamp").set_index("datetime")

        dfs[topic] = df

        out_csv = os.path.join(output_dir, topic.replace("/", "_") + ".csv")
        df.to_csv(out_csv)

    print("Merging topics...")

    df_all = functools.reduce(
        lambda l, r: pd.merge(l, r, how="outer", left_index=True, right_index=True),
        dfs.values(),
    )

    numeric_cols = df_all.select_dtypes(include=[np.number]).columns
    df_all[numeric_cols] = df_all[numeric_cols].interpolate(method="time")

    if SYNC_TOPIC in dfs:
        df_all = df_all.loc[df_all.index.intersection(dfs[SYNC_TOPIC].index)]

    df_all = df_all.dropna(subset=numeric_cols)

    out_file = os.path.join(output_dir, "topics_combined.csv")
    df_all.to_csv(out_file)

    print("Done:", bag_path)


def main():
    if not os.path.isdir(DATA_DIR):
        print(f"Directory not found: {DATA_DIR}")
        return

    bag_files = [
        os.path.join(DATA_DIR, f)
        for f in os.listdir(DATA_DIR)
        if f.endswith(".bag") and not f.endswith(".orig.bag")
    ]

    if not bag_files:
        print("No .bag files found.")
        return

    print(f"Found {len(bag_files)} bag files")

    for bag_path in sorted(bag_files):
        process_bag(bag_path)

    print("\nAll processing complete")


if __name__ == "__main__":
    main()
