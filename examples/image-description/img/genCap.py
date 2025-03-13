import json
import csv

with open('open_images_train_v6_captions.json', 'r') as json_file, \
     open('captions.csv', 'w', newline='', encoding='utf-8') as csv_file:
    
    csv_writer = csv.writer(csv_file)
    
    count = 0
    for line_number, json_line in enumerate(json_file, 1):
        try:
            parsed = json.loads(json_line.strip())  
            caption = parsed.get("caption", "")  
            csv_writer.writerow([caption])  
            
        except json.JSONDecodeError:
            print(f"警告：第{line_number}行JSON格式异常，已跳过")
        
        count += 1
        if count == 100:
            break
