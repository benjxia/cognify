import os

# Function to calculate averages of scores
def calculate_averages(score_list):
    if score_list:
        return sum(score_list) / len(score_list)
    return 0.0

# Initialize lists for the scores
bleu_scores = []
meteor_scores = []
cider_scores = []
combined_scores = []

# Check if the file exists and read previous scores if any
file_path = 'evaluation_scores.txt'
if os.path.exists(file_path):
    with open(file_path, 'r') as file:
        for line in file:
            # Parse the scores from the file
            if line.startswith("BLEU Score"):
                bleu_scores.append(float(line.split(":")[1].split("/")[0].strip()))
            elif line.startswith("METEOR Score"):
                meteor_scores.append(float(line.split(":")[1].split("/")[0].strip()))
            elif line.startswith("CIDEr Score"):
                cider_scores.append(float(line.split(":")[1].split("/")[0].strip()))


# Calculate the averages
avg_bleu = calculate_averages(bleu_scores[30:])
avg_meteor = calculate_averages(meteor_scores[30:])
avg_cider = calculate_averages(cider_scores[30:])
avg_combined = calculate_averages(combined_scores[30:])

# Append the averages to the file
with open("avg_evaluation_scores", 'a') as file:
    file.write(f"Average BLEU Score: {avg_bleu:.2f}/10\n")
    file.write(f"Average METEOR Score: {avg_meteor:.2f}/10\n")
    file.write(f"Average CIDEr Score: {avg_cider:.2f}/10\n")
    file.write("\n")