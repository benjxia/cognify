import csv

def generate_unique_labels(filtered_path, descriptions_path, output_path):
    label_dict = {}
    with open(descriptions_path, 'r', encoding='utf-8') as desc_file:
        reader = csv.reader(desc_file)
        for code, label in reader:  
            label_dict[code.strip()] = label.strip()

    unique_labels = set()
    with open(filtered_path, 'r', encoding='utf-8') as filtered_file:
        reader = csv.reader(filtered_file)
        for row in reader:
            if len(row) >= 2:
                mid = row[1].strip()
                if mid in label_dict:
                    unique_labels.add(label_dict[mid])

    sorted_labels = sorted(unique_labels)
    output_line = ','.join(sorted_labels)

    with open(output_path, 'w', encoding='utf-8') as out_file:
        out_file.write(output_line)

generate_unique_labels(
    filtered_path='filtered.csv',
    descriptions_path='class-descriptions-boxable.csv',
    output_path='unique_labels.txt'
)