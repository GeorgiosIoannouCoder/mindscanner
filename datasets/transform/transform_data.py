################################################################################################################

# Ioannou_Georgios and Yang_Zechen
# Copyright © 2025 by Georgios Ioannou and Zechen Yang
# Filename: transform_data.py
# Last Updated Date: 04/30/2025

################################################################################################################

# CDS NYU
# BMIN-GA 3007 | Deep Learning in Medicine
# Final Project
# Due Date: Monday, May 06, 2025, 11:59 PM

################################################################################################################

# Firstname: Georgios
# Lastname : Ioannou
# Group    : MindScanner
# Net ID   : gi2100
# Univ ID  : N16435765
# E-mail   : gi2100@nyu.edu

################################################################################################################

# Firstname: Zechen
# Lastname : Yang
# Group    : MindScanner
# Net ID   : zy3398
# Univ ID  : N12614073
# E-mail   : zy3398@nyu.edu


################################################################################################################
# Import libraries.
import os
import pandas as pd
import random

from nltk.corpus import wordnet
from tqdm import tqdm

################################################################################################################

# Set the seed for reproducibility.
random.seed(42)


################################################################################################################


def custom_transform(text):
    """
    Applies transformations to the text:
    1. Synonym replacement: Replaces some words with their synonyms
    2. Random typos: Simulates typing errors with keyboard adjacency
    """

    if not isinstance(text, str) or not text.strip():
        return text

    words = text.split()
    new_words = []

    # Define keyboard adjacency for common typos.
    keyboard_adjacency = {
        "a": ["s", "q", "z", "w"],
        "e": ["r", "d", "w", "s"],
        "i": ["u", "o", "k", "j"],
        "o": ["i", "p", "l", "k"],
        "u": ["y", "i", "j", "h"],
        "t": ["r", "y", "f", "g"],
        "s": ["a", "d", "w", "x"],
    }

    # Process each word.
    for word in words:
        # With 10% probability, try synonym replacement.
        if random.random() < 0.1 and len(word) > 3:
            synonyms = []
            try:
                for syn in wordnet.synsets(word):
                    for lemma in syn.lemmas():
                        if lemma.name() != word and "_" not in lemma.name():
                            synonyms.append(lemma.name())
                if synonyms:
                    new_words.append(random.choice(synonyms))
                else:
                    new_words.append(word)
            except:
                new_words.append(word)
        # With 10% probability, introduce a typo.
        elif random.random() < 0.1 and len(word) > 3:
            char_pos = random.randint(0, len(word) - 1)
            char = word[char_pos]
            if char in keyboard_adjacency:
                typo_char = random.choice(keyboard_adjacency[char])
                word = word[:char_pos] + typo_char + word[char_pos + 1 :]
            new_words.append(word)
        else:
            new_words.append(word)

    return " ".join(new_words)


################################################################################################################
def transform_dataset(input_file, output_dir):
    """
    Transform the entire dataset and save both original and transformed texts.
    """

    print(f"Reading dataset from {input_file}...")

    df = pd.read_csv(input_file)

    # Create output directory if it does not exist.
    os.makedirs(output_dir, exist_ok=True)

    # Only filter out empty texts, keep all valid posts.
    df["title_text_clean"] = df["title_text_clean"].fillna(
        ""
    )  # Replace NaN with empty string.
    df["subreddit"] = df["subreddit"].fillna("unknown")  # Replace NaN subreddits.

    # Remove completely empty texts but keep very short ones.
    df = df[df["title_text_clean"].str.len() > 0]

    print("\nOriginal dataset size:", len(df))
    print("\nClass distribution before transformation:")
    print(df["subreddit"].value_counts())

    # Create lists to store transformed texts.
    transformed_texts = []

    print("\nTransforming texts...")
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Applying transformations"):
        text = row["title_text_clean"]

        if isinstance(text, str):  # Additional check for string type.
            transformed = custom_transform(text)
            transformed_texts.append(transformed)

    # Add transformed text as a new column in the original DataFrame.
    df["title_text_clean_transform"] = transformed_texts

    # Save transformed dataset.
    output_file = os.path.join(output_dir, "transform_dataset.csv")
    df.to_csv(output_file, index=False)
    print(f"\nTransformed dataset saved to {output_file}")

    # Print statistics.
    print("\nDataset statistics:")
    print(f"Total samples processed: {len(df)}")
    print("\nSamples per subreddit:")
    print(df["subreddit"].value_counts())

    # Show a few examples.
    print("\nExample transformations:")
    for i in range(3):
        idx = random.randint(0, len(df) - 1)
        print(f"\nExample {i+1} ({df.iloc[idx]['subreddit']}):")
        print(f"Original Clean: {df.iloc[idx]['title_text_clean'][:200]}...")
        print(f"Transformed: {df.iloc[idx]['title_text_clean_transform'][:200]}...")


################################################################################################################
if __name__ == "__main__":
    transform_dataset(input_file="../clean/clean_dataset.csv", output_dir="./")
################################################################################################################
