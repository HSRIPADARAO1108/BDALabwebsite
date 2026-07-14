from collections import Counter

# Read input file
with open("input.txt", "r") as file:
    text = file.read().lower()

# Split into words
words = text.split()

# Count words
word_count = Counter(words)

# Display result
print("Word Count\n")
for word, count in word_count.items():
    print(word, ":", count)
