import re
import matplotlib.pyplot as plt

def parse_log_data(log_string):
    """
    Parses a log string to extract training step, tokens per second,
    peak active memory, peak allocated memory, and peak reserved memory.
    """
    steps = []
    tokens_per_second = []
    mem_active = []
    mem_alloc = []
    mem_reserved = []

    # Regex to find the required data
    # It looks for "Step [number]", "tokens_per_second_per_gpu:[number]",
    # "peak_memory_active:[number]", "peak_memory_alloc:[number]",
    # and "peak_memory_reserved:[number]".
    pattern = re.compile(
        r'Step (\d+).*?'
        r'tokens_per_second_per_gpu:([\d.]+).*?'
        r'peak_memory_active:([\d.]+).*?'
        r'peak_memory_alloc:([\d.]+).*?'
        r'peak_memory_reserved:([\d.]+)'
    )

    for line in log_string.strip().split('\n'):
        match = pattern.search(line)
        if match:
            # Extract and convert the matched groups to their appropriate types
            step = int(match.group(1))

            # Only process data for every 100 steps
            if step % 100 == 0:
                tokens_per_second.append(float(match.group(2)))
                mem_active.append(float(match.group(3)))
                mem_alloc.append(float(match.group(4)))
                mem_reserved.append(float(match.group(5)))
                steps.append(step)


    return steps, tokens_per_second, mem_active, mem_alloc, mem_reserved

def plot_memory_usage(steps, mem_active, mem_reserved):
    """
    Plots the peak active and reserved memory over training steps and saves the plot as a PNG file.
    """
    # Create figure and a set of subplots
    fig, ax1 = plt.subplots(figsize=(12, 8))

    # Plot memory usage on the y-axis
    ax1.set_xlabel('Training Step')
    ax1.set_ylabel('Memory Usage (GiB)')
    ax1.plot(steps, mem_active, label='Peak Memory Active', marker='x', linestyle='--', color='green')
    ax1.plot(steps, mem_reserved, label='Peak Memory Reserved', marker='^', linestyle=':', color='red')
    ax1.tick_params(axis='y')
    ax1.grid(True, which='both', linestyle='--', linewidth=0.5)

    # Add a title and combine legends
    plt.title('GPU Memory Usage During Training')
    fig.legend(loc="upper left", bbox_to_anchor=(0.1, 0.9))

    # Save the plot to a PNG file
    plt.savefig('memory_usage_plot.png')
    print("Plot saved to 'memory_usage_plot.png'")

if __name__ == '__main__':
    try:
        # Read data from the file
        with open('./log_1753183493.txt', 'r') as file:
            log_data = file.read()

        # Parse the data
        steps, tokens_per_second, mem_active, mem_alloc, mem_reserved = parse_log_data(log_data)

        # Plot the required data and save the figure
        plot_memory_usage(steps, mem_active, mem_reserved)
    except FileNotFoundError:
        print("Error: The file 'xxxx.txt' was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

