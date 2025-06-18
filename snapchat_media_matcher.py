import json
import os
import csv
import re

# CONFIG -- change these paths as needed
JSON_PATH = "/Volumes/M3GAN/snapchat exports/json/chat_history.json"
MEDIA_FOLDER = "/Volumes/M3GAN/snapchat exports/chat_media"
OUTPUT_CSV = "/Volumes/M3GAN/snapchat exports/snapchat_media_match.csv"

# Load JSON
with open(JSON_PATH, 'r') as f:
    chat_data = json.load(f)

# Build lookup table of media_id -> (conversation, sender)
media_lookup = {}
for convo, messages in chat_data.items():
    for msg in messages:
        media_ids = msg.get("Media IDs", "")
        if media_ids:
            for media_id in media_ids.split("|"):
                media_id_clean = media_id.strip().replace(" ", "")
                media_lookup[media_id_clean] = {
                    "conversation": convo,
                    "sender": msg.get("From"),
                    "is_sender": msg.get("IsSender")
                }

# Walk through media folder
results = []
for filename in os.listdir(MEDIA_FOLDER):
    if not filename.startswith("20"):
        continue  # skip anything not following Snapchat export naming
    
    match = re.search(r"b~(.*?)\.", filename)
    if not match:
        continue
    
    media_id_from_file = "b~" + match.group(1)

    data = media_lookup.get(media_id_from_file)
    if data:
        results.append({
            "filename": filename,
            "media_id": media_id_from_file,
            "sender": data["sender"],
            "conversation": data["conversation"],
            "is_sender": data["is_sender"]
        })
    else:
        results.append({
            "filename": filename,
            "media_id": media_id_from_file,
            "sender": "UNKNOWN",
            "conversation": "UNKNOWN",
            "is_sender": "UNKNOWN"
        })

# Write CSV
with open(OUTPUT_CSV, 'w', newline='') as csvfile:
    fieldnames = ["filename", "media_id", "sender", "conversation", "is_sender"]
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for row in results:
        writer.writerow(row)

print("Done! Results saved to:", OUTPUT_CSV)
