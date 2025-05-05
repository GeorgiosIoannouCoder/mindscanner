################################################################################################################

# Ioannou_Georgios
# Copyright © 2025 by Georgios Ioannou
# Filename: clean_data.py
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

# Import libraries.
import pandas as pd
import re
import nltk
import sys
from typing import Any

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.tokenize import word_tokenize

# Download required NLTK data.
nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("omw-1.4")
nltk.download("punkt")


################################################################################################################


class Cleaner:
    """
    Cleaner class responsible for text preprocessing including:
    lowercasing, removing stopwords, numbers, punctuation, URLs,
    lemmatizing, and stemming.
    """

    def __init__(self) -> None:
        self.stop_words = set(stopwords.words("english"))
        self.lemmatizer = WordNetLemmatizer()
        self.stemmer = PorterStemmer()

    def make_lowercase(self, input_string: str) -> str:
        return " ".join(word.lower() for word in input_string.split())

    def remove_stopwords(self, input_string: str) -> str:
        return " ".join(
            word for word in input_string.split() if word not in self.stop_words
        )

    def remove_numbers(self, input_string: str) -> str:
        return "".join(char for char in input_string if not char.isdigit())

    def remove_punctuation(self, input_string: str) -> str:
        input_string = re.sub(
            r"[%s]" % re.escape("""!"#$%&'()*+,،-./:;<=>؟?@[\]^_`{|}~"""),
            " ",
            input_string,
        )
        input_string = input_string.replace("؛", "")
        input_string = re.sub(r"\s+", " ", input_string)
        return input_string.strip()

    def remove_urls(self, input_string: str) -> str:
        return re.sub(r"https?://\S+|www\.\S+", "", input_string)

    def lemmatization(self, input_string: str) -> str:
        return " ".join(
            self.lemmatizer.lemmatize(word) for word in input_string.split()
        )

    def stem_words(self, input_string: str) -> str:
        words = word_tokenize(input_string)
        return " ".join(self.stemmer.stem(word) for word in words)

    def pipeline(self, input_string: Any) -> str:
        """
        Apply the full cleaning pipeline to a given string.
        Converts input to string if it's not already.
        """

        input_string = str(input_string)
        input_string = self.make_lowercase(input_string)
        input_string = self.remove_stopwords(input_string)
        input_string = self.remove_numbers(input_string)
        input_string = self.remove_punctuation(input_string)
        input_string = self.remove_urls(input_string)
        input_string = self.lemmatization(input_string)
        input_string = self.stem_words(input_string)
        return input_string


################################################################################################################


def process_csv(file_path: str) -> pd.DataFrame:
    """
    Load the dataset, clean the text, and return the cleaned DataFrame.
    Includes sanity checks, null checks, duplicate handling, and cleaning.
    """

    print("Loading dataset...")
    df = pd.read_csv(file_path)

    print("\nRemoving 'mentalhealth' subreddit entries...")
    df = df[df["Subreddit"] != "mentalhealth"]

    print("\nStandardizing column names...")
    df.columns = df.columns.str.lower()
    df["subreddit"] = df["subreddit"].str.lower()

    print("\nCombining 'title' and 'text' columns...")
    df["title"] = df["title"].fillna("")
    df["text"] = df["text"].fillna("")
    df["title_text"] = df["title"] + " " + df["text"]

    print("\nSTART - SANITY CHECK")
    print("--------------------")
    print("original_dataset_df.shape =", df.shape, "\n")
    print("------------------")
    print("END - SANITY CHECK")
    print("*" * 100)

    print("\nSTART - CHECKING FOR NULL VALUES COLUMNWISE")
    print("--------------------------------")
    print("original_dataset_df.isnull().sum() =")
    print(df.isnull().sum(), "\n")
    print("------------------------------")
    print("END - CHECKING FOR NULL VALUES COLUMNWISE")
    print("*" * 100)

    print("\nSTART - CHECKING FOR NULL VALUES IN THE WHOLE TRAIN PANDAS DATAFRAME")
    print("-------------------------------------------------------")
    print("original_dataset_df.isnull().sum().sum() = ", df.isnull().sum().sum(), "\n")
    print("------------------------------------------------------------------")
    print("END - CHECKING FOR NULL VALUES IN THE WHOLE TRAIN PANDAS DATAFRAME")
    print("*" * 100)

    print("\nSTART - REMOVING ROWS WHERE SUBREDDIT IS A NULL VALUE")
    print("-----------------------------------------------------")
    df = df.dropna(subset=["subreddit"])
    print("---------------------------------------------------")
    print("END - REMOVING ROWS WHERE SUBREDDIT IS A NULL VALUE")
    print("*" * 100)

    print("\nSTART - REMOVING ROWS WHERE title_text IS A NULL VALUE")
    print("-----------------------------------------------------")
    df = df.dropna(subset=["title_text"])
    print("---------------------------------------------------")
    print("END - REMOVING ROWS WHERE title_text IS A NULL VALUE")
    print("*" * 100)

    print("\nSTART - CHECKING FOR DUPLICATE title_text")
    print("------------------------------------")
    duplicates = df.duplicated(subset="title_text")
    print("len(original_dataset_df[duplicates]) =", len(df[duplicates]))
    print(df[duplicates])
    df = df.drop_duplicates(subset="title_text")
    print("----------------------------------")
    print("END - CHECKING FOR DUPLICATE title_text")
    print("*" * 100)

    print("\nSTART - CHECKING FOR DUPLICATE ROWS")
    print("-----------------------------------")
    duplicates = df.duplicated()
    print("duplicates.sum() =", duplicates.sum())
    print("Duplicate Rows   =", df[duplicates])
    print("---------------------------------")
    print("END - CHECKING FOR DUPLICATE ROWS")
    print("*" * 100)

    print("\nSTART - REMOVING DUPLICATE ROWS")
    print("-------------------------------")
    df = df.drop_duplicates()
    print("-----------------------------")
    print("END - REMOVING DUPLICATE ROWS")
    print("*" * 100)

    print("\nSTART - DISPLAY CLASSES AND VALUE COUNTS")
    print("----------------------------------------")
    print("Unique subreddits =", sorted(df["subreddit"].unique()), "\n")
    print("# of Unique subreddits =", len(df["subreddit"].unique()), "\n")
    print(df.subreddit.value_counts(), "\n")
    print("--------------------------------------")
    print("END - DISPLAY CLASSES AND VALUE COUNTS")
    print("*" * 100)

    print("\nCleaning text with pipeline...")
    cleaner = Cleaner()
    df["title_text_clean"] = df["title_text"].apply(cleaner.pipeline)

    return df


################################################################################################################


def show_examples(df: pd.DataFrame, n: int = 3) -> None:
    """
    Print a few original and cleaned text examples.
    """
    print(f"\nShowing {n} examples of original vs cleaned text:\n")
    for i in range(min(n, len(df))):
        print(f"--- Example {i + 1} ---")
        print("Original:\n", df.iloc[i]["title_text"])
        print("\nCleaned:\n", df.iloc[i]["title_text_clean"])
        print("\n")


################################################################################################################


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 clean_data.py path/to/original_dataset.csv")
        sys.exit(1)

    csv_path = sys.argv[1]
    cleaned_df = process_csv(csv_path)

    output_path = "clean_dataset.csv"
    print(f"\nSaving cleaned dataset to {output_path}...")
    cleaned_df.to_csv(output_path, index=False)

    show_examples(cleaned_df)
    print("\nDone.")
################################################################################################################
