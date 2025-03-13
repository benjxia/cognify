import json
import csv

with open('open_images_train_v6_captions.json', 'r') as json_file, \
     open('ids.csv', 'w', newline='', encoding='utf-8') as csv_file:
    
    writer = csv.writer(csv_file)
    count = 0
    for line in json_file:
        data = json.loads(line.strip())  # [1,6](@ref)
        
        image_id = data.get("image_id", "")  # [6,7](@ref)
        
        writer.writerow(['train/'+image_id])  # [9,10](@ref)

        count += 1
        if count == 100:
            break