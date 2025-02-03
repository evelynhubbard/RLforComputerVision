FINALPROJECT/      # Directory for storing model checkpoints
├── Code/                      # Main project codebase
│   ├── configs/               # Configuration files
│   │   └── tiny-imageNet.yml  # YAML file for Tiny ImageNet configurations
│   ├── data/                  # Dataset-related files
│   │   ├── processed/         # Processed datasets (ready for use)
│   │   ├── raw/               # Raw datasets (original structure)
│   ├── datasets/              # Dataset utilities and loaders
│   │   ├── __init__.py        # Makes `datasets` an importable module
│   │   └── tiny_imagenet.py   # Functions for Tiny ImageNet restructuring, loading, summarization
│   ├── functions/             # General-purpose utility functions
│   │   ├── __init__.py        # Makes `functions` an importable module
│   │   ├── feature_extraction.py  # Feature extraction utilities
│   │   ├── q_helper.py        # Helper function for Q-learning
│   │   ├── utils.py           # Utils
│   │   └── visualization.py   # Functions for data visualization
│   ├── models/                # Model definitions
│   │   ├── __pycache__/       # Cached Python files (auto-generated, ignore)
│   │   ├── __init__.py        # Makes `models` an importable module
│   │   ├── q_classifier.py    # Q-Learning definition
│   │   ├── resnet.py          # Custom ResNet50 CNN model definition
│   │   └── secondary_classifier.py  # Secondary classifier definition
│   │   └── secondary_classifier_NN/  # Saved trained secondary classifier model weights
│   │   └── resnet_model/       # Saved trained CNN weights
│   ├── results/               # Results from training and evaluation
│   │   └── ImageNet/10epochs/ # Resultsfrom10epochs
│   ├── runners/               # Scripts for running workflows
│   │   ├── __pycache__/       # Cached Python files (auto-generated, ignore)
│   │   ├── __init__.py        # Makes `runners` an importable module
│   │   ├── train_cnn.py       # Trains the CNN, secondary_classifier and Q-learning models
│   │   └── test.py            # Tests the accuracy of the deep models with and without Q-Learning
│   ├── main.py                # Entry point for the project
│   └── README.txt             # Documentation
└── requirements.txt           # Python dependencies
└── ImageClassification by RL.pdf  # Paper by Hafiz, which the implementation is aiming to replicate
└── report_EH.pdf

