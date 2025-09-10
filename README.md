# 🎓 Machine Learning Assignment: Comprehensive Implementation

![Python](https://img.shields.io/badge/Python-3.x-blue?style=for-the-badge&logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-Neural_Networks-red?style=for-the-badge&logo=pytorch)
![Academic](https://img.shields.io/badge/Academic-Assignment-green?style=for-the-badge&logo=graduation-cap)
![Machine Learning](https://img.shields.io/badge/Machine-Learning-orange?style=for-the-badge&logo=brain)

A comprehensive machine learning assignment demonstrating proficiency across diverse ML paradigms, from classical probabilistic modeling to modern deep learning architectures. This academic project showcases the integration of statistical analysis, neural networks, and dimensionality reduction techniques.

## 🎯 Academic Overview

This assignment represents a comprehensive exploration of machine learning methodologies, designed to demonstrate mastery across both traditional statistical approaches and cutting-edge deep learning techniques. The project is structured as a two-part academic assignment, each worth 50 marks, covering the breadth of modern machine learning.

**Educational Objective**: Demonstrate practical implementation skills across diverse machine learning paradigms while maintaining academic rigor and mathematical foundation.

## 📚 Assignment Structure

### Part 1: Probabilistic Modeling & Statistical Analysis 
**Focus**: Classical machine learning and statistical modeling
- Probabilistic tracking and prediction systems
- Gaussian noise modeling and likelihood estimation
- Mathematical modeling for spatial-temporal data
- Vector-based directional analysis

### Part 2: Neural Networks, Dimensionality Reduction & Clustering 
**Focus**: Modern deep learning and neural network architectures
- Convolutional Neural Networks (CNNs)
- Fully Connected Neural Networks (FCNNs)
- Denoising Autoencoders
- Dimensionality reduction and clustering techniques

## ✨ Key Learning Objectives

### 🧮 Mathematical Foundation
- **Probabilistic Modeling** - Gaussian distributions and likelihood functions
- **Statistical Analysis** - Negative log-likelihood and uncertainty quantification
- **Vector Mathematics** - Directional analysis and spatial computations
- **Linear Algebra** - Dimensionality reduction and feature spaces

### 🤖 Deep Learning Proficiency
- **Neural Network Architectures** - CNN, FCNN, and autoencoder design
- **PyTorch Implementation** - Modern deep learning framework usage
- **Training Optimization** - Model training and hyperparameter tuning
- **Model Evaluation** - Performance assessment and validation

### 📊 Data Science Skills
- **Feature Engineering** - Data preprocessing and transformation
- **Dimensionality Reduction** - PCA and autoencoder techniques
- **Clustering Analysis** - Unsupervised learning methods
- **Model Persistence** - Saving and loading trained models

## 🔬 Part 1: Probabilistic Modeling Implementation

### Problem Domain: Bee Flight Path Reconstruction
```python
# Core mathematical framework
def negative_log_likelihood(prediction, observation, sigma):
    """
    Calculate negative log-likelihood for spatial prediction
    
    Args:
        prediction: Predicted bee location
        observation: Detector observation data
        sigma: Gaussian noise parameter
    
    Returns:
        Negative log-likelihood value
    """
    error_vector = prediction - observation
    likelihood = np.exp(-np.linalg.norm(error_vector)**2 / (2 * sigma**2))
    return -np.log(likelihood)
```

### Technical Implementation
- **Dataset**: 30 bee flight paths with 100 spatial points over 30 seconds
- **Observations**: 17 detector measurements per flight path
- **Methodology**: Gaussian noise model with directional bearing analysis
- **Output**: Probabilistic reconstruction of movement patterns

### Key Features
- **Vector-based Analysis** - Unit vector calculations for direction
- **Gaussian Modeling** - Statistical noise representation
- **Spatial Tracking** - Multi-detector location triangulation
- **Uncertainty Quantification** - Likelihood-based confidence estimation

## 🧠 Part 2: Neural Networks & Deep Learning

### Model Architectures Implemented

#### 1. Convolutional Neural Network (CNN)
```python
# CNN Architecture (best_model_CNN.pth)
class CNN_Classifier(nn.Module):
    def __init__(self):
        super(CNN_Classifier, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(feature_size, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )
```

#### 2. Fully Connected Neural Network (FCNN)
```python
# FCNN Architecture (best_model_FCNN.pth)
class FCNN_Classifier(nn.Module):
    def __init__(self, input_size, hidden_sizes, num_classes):
        super(FCNN_Classifier, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_sizes[0]),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_sizes[0]),
            nn.Dropout(0.3),
            nn.Linear(hidden_sizes[0], hidden_sizes[1]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[1], num_classes)
        )
```

#### 3. Denoising Autoencoder
```python
# Autoencoder Architecture (denoising_autoencoder.pth)
class DenoisingAutoencoder(nn.Module):
    def __init__(self, input_dim, encoding_dim):
        super(DenoisingAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, encoding_dim * 2),
            nn.ReLU(),
            nn.Linear(encoding_dim * 2, encoding_dim),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, encoding_dim * 2),
            nn.ReLU(),
            nn.Linear(encoding_dim * 2, input_dim),
            nn.Sigmoid()
        )
```

### Technical Achievements
- **Multi-Architecture Comparison** - Performance evaluation across CNN, FCNN, and autoencoder
- **Hyperparameter Optimization** - Systematic tuning for optimal performance
- **Model Persistence** - Professional model saving and loading procedures
- **Dimensionality Reduction** - Autoencoder-based feature compression

## 🛠️ Technologies & Frameworks

### Core Framework
- **Python 3.x** - Primary programming language
- **PyTorch** - Deep learning framework for neural networks
- **NumPy** - Numerical computations and array operations
- **Pandas** - Data manipulation and analysis

### Machine Learning Libraries
- **Scikit-learn** - Classical ML algorithms and utilities
- **Matplotlib** - Data visualization and plotting
- **Seaborn** - Statistical data visualization
- **Jupyter Notebook** - Interactive development environment

### Mathematical Libraries
- **SciPy** - Scientific computing and statistical functions
- **Math** - Mathematical functions and constants
- **Statistics** - Statistical analysis tools

## 🚀 Installation & Setup

### Prerequisites
```bash
Python 3.7+
Jupyter Notebook
pip package manager
```

### Quick Setup
```bash
# Clone the repository
git clone https://github.com/Zyttik-m/Assignment-Machine-Learning.git
cd Assignment-Machine-Learning

# Install required dependencies
pip install -r requirements.txt

# Launch Jupyter Notebook
jupyter notebook assignment_acp24kc.ipynb
```

### Dependencies
```bash
pip install torch torchvision numpy pandas matplotlib seaborn
pip install scikit-learn scipy jupyter notebook
```

## 💻 Usage Guide

### Running the Complete Assignment
```python
# Open the Jupyter notebook
jupyter notebook assignment_acp24kc.ipynb

# Execute cells sequentially:
# 1. Part 1: Probabilistic modeling and bee tracking
# 2. Part 2: Neural network implementations
# 3. Model training and evaluation
# 4. Results analysis and comparison
```

### Model Loading and Evaluation
```python
import torch

# Load trained models
cnn_model = torch.load('best_model_CNN.pth')
fcnn_model = torch.load('best_model_FCNN.pth')
autoencoder = torch.load('denoising_autoencoder.pth')

# Evaluate model performance
cnn_model.eval()
with torch.no_grad():
    predictions = cnn_model(test_data)
```

## 📁 Project Structure

```
Assignment-Machine-Learning/
│
├── 📓 Main Assignment
│   └── assignment_acp24kc.ipynb    # Complete assignment notebook
│       ├── Part 1: Probabilistic Modeling
│       │   ├── Bee tracking implementation
│       │   ├── Gaussian noise modeling
│       │   ├── Likelihood calculations
│       │   └── Statistical analysis
│       │
│       └── Part 2: Neural Networks
│           ├── CNN implementation
│           ├── FCNN architecture
│           ├── Autoencoder design
│           └── Clustering analysis
│
├── 🤖 Trained Models
│   ├── best_model_CNN.pth           # Optimized CNN model
│   ├── best_model_FCNN.pth          # Best FCNN architecture
│   └── denoising_autoencoder.pth    # Trained autoencoder
│
└── 📊 Supporting Files
    ├── data/                        # Dataset files
    ├── results/                     # Output analyses
    └── utils/                       # Helper functions
```

## 📈 Academic Results & Analysis

### Part 1: Probabilistic Modeling Performance
- **Accuracy**: High-precision spatial tracking using statistical methods
- **Innovation**: Vector-based directional analysis with Gaussian uncertainty
- **Mathematical Rigor**: Proper likelihood estimation and error quantification
- **Practical Application**: Real-world animal movement tracking

### Part 2: Neural Network Comparison
| Model | Architecture | Performance | Use Case |
|-------|-------------|-------------|----------|
| CNN | Convolutional | High accuracy on spatial data | Image classification |
| FCNN | Fully Connected | Robust general classification | Tabular data |
| Autoencoder | Encoder-Decoder | Effective dimensionality reduction | Feature learning |

### Key Learning Outcomes
- **Classical vs Modern ML** - Understanding when to apply different approaches
- **Mathematical Foundation** - Strong statistical and probabilistic modeling
- **Deep Learning Proficiency** - PyTorch implementation and optimization
- **Academic Rigor** - Systematic approach to ML problem-solving

## 🔬 Technical Innovations

### Part 1 Contributions
- **Novel Tracking Algorithm** - Probabilistic reconstruction of movement patterns
- **Multi-Detector Integration** - Spatial triangulation with uncertainty quantification
- **Mathematical Modeling** - Gaussian noise representation for real-world data

### Part 2 Achievements
- **Architecture Optimization** - Comparative analysis of neural network designs
- **Autoencoder Innovation** - Denoising capabilities for robust feature learning
- **Model Persistence** - Professional model management and deployment

### Learning Objectives Met
- ✅ **Statistical Modeling** - Probabilistic analysis and likelihood estimation
- ✅ **Deep Learning** - Neural network design and implementation
- ✅ **Mathematical Rigor** - Proper mathematical foundation and analysis
- ✅ **Practical Implementation** - Real-world problem-solving applications
- ✅ **Technical Communication** - Clear documentation and analysis

### Academic Standards
- **Individual Work** - Original implementation and analysis
- **Code Quality** - Clean, well-documented Python code
- **Mathematical Accuracy** - Correct implementation of statistical methods
- **Critical Analysis** - Thoughtful evaluation of different approaches

## 🔮 Educational Extensions

### Advanced Topics Covered
- **Bayesian Inference** - Probabilistic reasoning and uncertainty
- **Deep Learning Theory** - Neural network mathematical foundations
- **Optimization Techniques** - Gradient descent and backpropagation
- **Model Evaluation** - Validation and performance assessment

### Skills Development
- **Research Methodology** - Systematic approach to ML problems
- **Technical Writing** - Clear documentation and analysis
- **Problem Decomposition** - Breaking complex problems into manageable parts
- **Critical Thinking** - Evaluating different approaches and methodologies

## 📚 Academic References

This assignment incorporates established ML concepts from:

- **Bishop, C.M.** (2006). Pattern Recognition and Machine Learning
- **Goodfellow, I. et al.** (2016). Deep Learning
- **Murphy, K.P.** (2012). Machine Learning: A Probabilistic Perspective
- **Hastie, T. et al.** (2009). The Elements of Statistical Learning

## 🏆 Academic Achievement

### Demonstrated Competencies
- **Mathematical Modeling** - Advanced statistical and probabilistic analysis
- **Deep Learning Proficiency** - Modern neural network implementation
- **Programming Excellence** - Clean, efficient Python and PyTorch code
- **Analytical Thinking** - Critical evaluation and comparison of methods
- **Technical Communication** - Clear documentation and result presentation

### Professional Skills
- **Research Methodology** - Systematic approach to complex problems
- **Technical Implementation** - Practical ML solution development
- **Performance Optimization** - Model tuning and improvement techniques
- **Academic Writing** - Professional documentation and analysis

## 👨‍🎓 Author

**Kittithat Chalermvisutkul**
- **Academic Background**: MSc Computer Science (Artificial Intelligence)
- **GitHub**: [@Zyttik-m](https://github.com/Zyttik-m)
- **LinkedIn**: [linkedin.com/in/Kittithat-CH](https://linkedin.com/in/Kittithat-CH)



---

🎓 **Academic Excellence in Machine Learning Implementation**  
📚 **Demonstrating Comprehensive ML Proficiency from Classical to Modern Techniques**

![Academic Achievement](https://via.placeholder.com/800x400/4169e1/ffffff?text=Machine+Learning+Assignment+Results)

*Dedicated to advancing machine learning education and practical implementation skills*
