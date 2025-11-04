import re
import matplotlib.pyplot as plt

# File path to your log
log_file_path = "./log_1756467542.txt"

# Read the log file
with open(log_file_path, "r") as f:
    log_text = f.read()

# Extract step and throughput using regex
pattern = r"Step (\d+).*?tokens_per_second_per_gpu:([\d\.]+)"
steps = []
throughput = []

for match in re.finditer(pattern, log_text):
    steps.append(int(match.group(1)))
    throughput.append(float(match.group(2)))

# Plotting
plt.figure(figsize=(10, 5))
plt.plot(steps, throughput, marker='o', linestyle='-', color='teal')
plt.title("Tokens/sec/GPU vs Training Step")
plt.xlabel("Training Step")
plt.ylabel("Tokens/sec/GPU")
plt.grid(True)
plt.tight_layout()

# Save the figure to a file
# You can change "throughput_plot.png" to any name you like (e.g., .jpg, .pdf)
plt.savefig("throughput_plot.png")

# If you also want to see the plot after saving, uncomment the next line
plt.show()
