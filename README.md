# Data Deduplication for Large Language Model Training

This repository provides a script for deduplicating text data, which is essential for training more efficient and effective LLM. The deduplication process reduces redundancy, which can improve the generalization ability of models trained on the dataset.

## Overview

Data deduplication is crucial in preparing high-quality datasets for training machine learning models, particularly for NLP tasks. This script uses both lexical and semantic analysis to identify and remove duplicate entries from datasets.

- **Lexical Deduplication**: Identifies duplicates by comparing the syntactic structure of the text.
- **Semantic Deduplication**: Uses Sentence Transformers to understand the meaning behind texts and identify duplicates even when the wording is not exactly the same.

## Getting Started

### Prerequisites

Before you run the script, ensure you have the following installed:

- Python 3.6+
### Installation

1. **Clone the Repository**

   ```bash
   git clone https://github.com/Jay-Nehra/dataset-deduplication.git
   cd dataset-deduplication
   ```

2. **Install Dependencies**

   Install all required libraries using pip:

   ```bash
   pip install pandas numpy faiss-cpu sentence-transformers datasketch tqdm loguru pyyaml nltk
   ```

### Configuration

Edit the `config.yaml` file to match your dataset paths and preferences. Here is an example configuration:

```yaml
settings:
  input_directory: "./source"
  output_directory: "./output"
  jaccard_threshold: 0.7
  num_permutations: 512
  semantic_deduplication_threshold: 0.8

categories:
  - name: "healthcare"
    datasets:
      - name: "Healthcare Dataset"
        file_name: "healthcare.csv"
        deduplication_column: "text"
  - name: "psychology"
    datasets:
      - name: "Psychology Dataset"
        file_name: "psychology.csv"
        deduplication_column: "text"
```

## Usage

To run the deduplication process, use the following command in the project's root directory:

```bash
python3 main.py
```

### Outputs

After running the script, the deduplicated and pruned datasets will be found in the `output` directory:

- `./output/De_duplicated/`: Contains the deduplicated datasets.
- `./output/Pruned/`: Contains the records that were identified as duplicates and the corresponding record that was kept.

### Directory Structure

```
.
├── config.yaml                 # Configuration file to set parameters for deduplication
├── logs                        # Directory where logs are stored
│   └── deduplication_log_{timestamp}.log # Example log file
├── main.py                     # Main Python script for deduplication
├── output                      # Directory where the output files are stored
│   ├── De_duplicated           # Contains deduplicated datasets
│   │   ├── deduped_healthcare.csv
│   │   └── deduped_psychology.csv
│   └── Pruned                  # Contains records pruned during deduplication
│       ├── pruned_healthcare.csv
│       └── pruned_psychology.csv
└── source                      # Directory containing the original datasets
    ├── healthcare.csv
    └── psychology.csv
```

## Contributing

Any contributions you make are **greatly appreciated**.

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request



## Contact

Jayant Nehra - [@LinkedIn](https://www.linkedin.com/in/jayant-nehra/) - [@Gmail](nj.nehra@gmail.com)

Project Link: [Text Based Dataset De-Duplication](https://github.com/Jay-Nehra/dataset-deduplication)

