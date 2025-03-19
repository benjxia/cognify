for i in $(seq 1 5); do
    cognify optimize -f workflow.py >> "ImageQuality test VLM.txt"
done