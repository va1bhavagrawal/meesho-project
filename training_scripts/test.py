import argparse

# Create the ArgumentParser object
parser = argparse.ArgumentParser()

# Define the arguments
parser.add_argument("--name", type=str, default="John Doe")
parser.add_argument("--age", type=int, default=30)

# Parse the arguments
args = parser.parse_args()

# Manually set the value of an argument
args.name = "Jane Smith"

# Access the values of the arguments
print(f"Name: {args.name}")
print(f"Age: {args.age}")