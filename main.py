import os
import re
import pandas as pd
import yaml
from datasketch import MinHash, MinHashLSH
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sentence_transformers import SentenceTransformer
import numpy as np
from tqdm import tqdm
from loguru import logger
import faiss
from icecream import ic

nltk.download('stopwords', quiet=True)
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()
NON_ALPHA_REGEX = re.compile("[^A-Za-z0-9_ ]")
MIN_NUM_TOKENS = 1
NUM_PERM = 512

def load_config(file_path):
    """
    Load the configuration from a YAML file.

    Args:
        file_path (str): The path to the YAML file.

    Returns:
        dict: The loaded configuration as a dictionary.
    """
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)

# Text preprocessing
def tokenize_text(text):
    """
    Tokenizes the given text by removing non-alphabetic characters, converting the text to lowercase, and stemming each word.

    Parameters:
        text (str): The text to be tokenized.

    Returns:
        List[str]: A list of stemmed words from the text, excluding stop words.
    """
    if pd.isna(text):
        return []
    text = NON_ALPHA_REGEX.sub(' ', text.lower())
    return [stemmer.stem(token) for token in text.split() ] # if token not in stop_words

def calculate_min_hash(tokens, num_perm):
    """
    Calculate the MinHash value for a set of tokens.

    Parameters:
        tokens (list): A list of tokens to calculate the MinHash value.
        num_perm (int): The number of permutations for the MinHash.

    Returns:
        MinHash: The MinHash object representing the set of tokens.
    """
    if not tokens:
        return None
    min_hash = MinHash(num_perm=num_perm)
    for token in set(tokens):
        min_hash.update(token.encode('utf-8'))
    return min_hash

class DeduplicationEngine:
    def __init__(self, config):
        self.config = config
        self.model = SentenceTransformer('all-MiniLM-L6-v2')

    def deduplicate(self, df, text_column):
        """
        Deduplicates the input DataFrame based on the text_column provided. 
        Performs both lexical and semantic deduplication using MinHashLSH and FAISS.
        
        Parameters:
            df (DataFrame): The input DataFrame containing the text data.
            text_column (str): The column name in the DataFrame that contains the text to deduplicate.
        
        Returns:
            Tuple[DataFrame, DataFrame]: Two DataFrames, one with deduplicated records and one with pruned records.
        """
        lsh = MinHashLSH(threshold=self.config['settings']['jaccard_threshold'], num_perm=NUM_PERM)
        min_hashes = {}
        all_kept_indices = set(df.index)  

        for idx, text in tqdm(df[text_column].items(), total=len(df), desc="Lexical deduplication"):
            tokens = tokenize_text(text)
            min_hash = calculate_min_hash(tokens, NUM_PERM)
            if min_hash:
                lsh.insert(str(idx), min_hash)
                min_hashes[idx] = min_hash

        processed = set()
        clusters = []
        for idx, min_hash in min_hashes.items():
            if idx in processed:
                continue
            result = lsh.query(min_hash)
            if len(result) > 1:
                cluster = sorted(map(int, result))
                clusters.append(cluster)
                processed.update(cluster)

        pruned_records = []

        for cluster in tqdm(clusters, desc="Semantic deduplication"):
            if len(cluster) <= 1:
                continue

            texts = df.loc[cluster, text_column].tolist()
            embeddings = self.model.encode(texts)

            dimension = embeddings.shape[1]
            index = faiss.IndexFlatL2(dimension)
            faiss.normalize_L2(embeddings)
            index.add(embeddings)

            _, I = index.search(embeddings, 2)  

            best_index = np.argmin(I[:, 1])
            kept_index = cluster[best_index]
            all_kept_indices.update([kept_index])  

            for pruned_idx in cluster:
                if pruned_idx != kept_index:
                    pruned_records.append({
                        text_column: df.at[pruned_idx, text_column],
                        'Kept_Record': df.at[kept_index, text_column]
                    })
                    all_kept_indices.discard(pruned_idx)  

        deduped_df = df.loc[sorted(all_kept_indices)].reset_index(drop=True)
        pruned_df = pd.DataFrame(pruned_records)

        return deduped_df, pruned_df

def main():
    """
    Generates the main function that performs data deduplication based on the provided configuration.

    Returns:
        None
    """
    config = load_config('config.yaml')
    deduplicator = DeduplicationEngine(config)

    output_dir = config['settings']['output_directory']
    os.makedirs(output_dir, exist_ok=True)

    dedup_dir = os.path.join(output_dir, "De_duplicated")
    pruned_dir = os.path.join(output_dir, "Pruned")
    os.makedirs(dedup_dir, exist_ok=True)
    os.makedirs(pruned_dir, exist_ok=True)

    for category in config['categories']:
        for dataset in category['datasets']:
            input_path = os.path.join(config['settings']['input_directory'], dataset['file_name'])
            if not os.path.exists(input_path):
                logger.error(f"File not found: {input_path}")
                continue

            df = pd.read_csv(input_path)
            if df.empty:
                logger.warning(f"No data to process in {input_path}.")
                continue

            logger.info(f"Starting deduplication for {dataset['file_name']} with {len(df)} records.")
            final_df, pruned_df = deduplicator.deduplicate(df, dataset['deduplication_column'])
            logger.info(f"Deduplication complete for {dataset['file_name']}. {len(final_df)} records kept, {len(pruned_df)} records pruned.")

            deduped_path = os.path.join(dedup_dir, f"deduped_{dataset['file_name']}")
            pruned_path = os.path.join(pruned_dir, f"pruned_{dataset['file_name']}")

            final_df.to_csv(deduped_path, index=False)
            pruned_df.to_csv(pruned_path, index=False)

            ic(deduped_path)
            ic(pruned_path)

if __name__ == "__main__":
    logger.add(os.path.join('logs', 'deduplication_log_{time}.log'), rotation="10 MB")
    main()
