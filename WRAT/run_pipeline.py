import sys
import subprocess
import argparse
import re

# Ensure Windows terminal doesn't crash on standard unicode output
if sys.stdout.encoding.lower() not in ['utf-8', 'utf8']:
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except AttributeError:
        pass

def draw_progress_bar(current, total):
    percent = float(current) * 100 / total
    print(f'Progress: {percent:.1f}% ({current}/{total} Epochs)')

def main():
    parser = argparse.ArgumentParser(description="WRAT Training Pipeline with Progress Tracking")
    parser.add_argument('--epochs', type=int, default=30, help="Number of epochs to pipeline")
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--seq_len', type=int, default=128)
    parser.add_argument('--lr', type=float, default=1e-3)

    args = parser.parse_args()

    # Pass the arguments down to main.py
    cmd = [
        sys.executable, "-u", "main.py", 
        "--epochs", str(args.epochs),
        "--batch_size", str(args.batch_size),
        "--seq_len", str(args.seq_len),
        "--lr", str(args.lr)
    ]

    print(f"Starting WRAT execution pipeline...")
    print(f"Command: {' '.join(cmd)}\n")

    # Start the subprocess, capturing stdout and stderr
    # universal_newlines=True ensures stdout is treated as a string with text encoding, rather than raw bytes
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True, encoding='cp1252', errors='replace')

    # Regex to hook onto the epoch progression
    # Assuming `main.py` logs exactly something like: ` 15 | τ=0.10 ... `
    epoch_pattern = re.compile(r"^\s*(\d+)\s*\|")

    output_lines = []

    try:
        # Initial empty progress bar
        draw_progress_bar(0, args.epochs)

        while True:
            # yield line by line
            line = process.stdout.readline()
            if not line:
                break
            
            output_lines.append(line)

            # Search if the current line describes epoch progression
            match = epoch_pattern.search(line)
            if match:
                current_epoch = int(match.group(1))
                draw_progress_bar(current_epoch, args.epochs)
            
            # Print the line if it is a benchmark output separator or hardware selection
            if line.startswith("TEST SET BENCHMARK") or line.startswith("WRAT wins") or line.startswith("Using device:"):
                # Clear progress line to avoid glitchy terminal wrapping
                sys.stdout.write('\r\n' + line)

    except KeyboardInterrupt:
        print("\nPipeline interrupted by user.")
        process.terminate()
        sys.exit(1)

    process.wait()
    print("\n\nWRAT Execution Pipeline Completed!")

    if process.returncode != 0:
        print(f"Warning: Encountered an error (exit code {process.returncode}). Check logs below if missing output.")
        # If there was an error, print the tail of the log
        print("".join(output_lines[-10:]))

if __name__ == "__main__":
    main()
