FINALPROJECT/
├── 626FinalProject_venv/      # Python virtual environment
├── checkpoints/               # Directory for storing model checkpoints
├── Code/                      # Main project codebase
│   ├── configs/               # Configuration files
│   │   └── tiny-imageNet.yml  # YAML file for Tiny ImageNet configurations
│   ├── data/                  # Dataset-related files
│   │   ├── processed/         # Processed datasets (ready for use)
│   │   ├── raw/               # Raw datasets (original structure)
│   │   └── __init__.py        # Makes `data` an importable module
│   ├── datasets/              # Dataset utilities and loaders
│   │   ├── __init__.py        # Makes `datasets` an importable module
│   │   └── tiny_imagenet.py   # Functions for Tiny ImageNet restructuring, loading, summarization
│   ├── functions/             # General-purpose utility functions
│   │   ├── __init__.py        # Makes `functions` an importable module
│   │   ├── feature_extraction.py  # Feature extraction utilities
│   │   ├── losses.py          # Custom loss functions
│   │   ├── metrics.py         # Custom metrics for evaluation
│   │   └── visualization.py   # Functions for data visualization
│   ├── log/                   # Logs for training and debugging
│   ├── models/                # Model definitions
│   │   ├── __pycache__/       # Cached Python files (auto-generated, ignore)
│   │   ├── __init__.py        # Makes `models` an importable module
│   │   ├── q_network.py       # Q-Network implementation
│   │   ├── resnet.py          # ResNet model definition
│   │   └── secondary_classifier.py  # Secondary classifier definition
│   ├── results/               # Results from training and evaluation
│   ├── runners/               # Scripts for running workflows
│   │   ├── __pycache__/       # Cached Python files (auto-generated, ignore)
│   │   ├── __init__.py        # Makes `runners` an importable module
│   │   ├── evaluate.py        # Evaluation workflow
│   │   ├── train_cnn.py       # Script for training the primary CNN
│   │   └── train_secondary.py # Script for training the secondary classifier
│   ├── utils/                 # Miscellaneous utilities
│   ├── main.py                # Entry point for the project
│   └── README.txt             # Documentation
└── requirements.txt           # Python dependencies
