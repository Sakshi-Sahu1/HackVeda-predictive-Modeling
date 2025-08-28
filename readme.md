# Student Performance Predictive Modeling

## Project Overview
This project builds predictive models to predict student math scores based on various demographic and educational factors using the Student Performance dataset from Kaggle.

## Dataset
**Source:** [Students Performance in Exams - Kaggle](https://www.kaggle.com/spscientist/students-performance-in-exams)

**Features:**
- Gender
- Race/ethnicity
- Parental level of education
- Lunch type (standard/free or reduced)
- Test preparation course completion
- Math score (target variable)
- Reading score
- Writing score

## Project Structure
```
student-performance-prediction/
│
├── data/
│   ├── raw/                    # Original dataset
│   ├── processed/              # Cleaned and preprocessed data
│   └── external/               # Additional data sources
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_data_preprocessing.ipynb
│   ├── 03_model_development.ipynb
│   └── 04_model_evaluation.ipynb
│
├── src/
│   ├── __init__.py
│   ├── data_preprocessing.py
│   ├── feature_engineering.py
│   ├── model_training.py
│   └── evaluation.py
│
├── models/                     # Saved model files
├── reports/                    # Analysis reports and visualizations
├── requirements.txt
├── .gitignore
├── README.md
└── LICENSE
```

## Installation & Setup

### Prerequisites
- Python 3.8+
- Jupyter Notebook
- Git

### Installation Steps
1. Clone the repository:
```bash
git clone https://github.com/yourusername/student-performance-prediction.git
cd student-performance-prediction
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install required packages:
```bash
pip install -r requirements.txt
```

4. Download the dataset:
   - Visit [Kaggle Dataset](https://www.kaggle.com/spscientist/students-performance-in-exams)
   - Download `StudentsPerformance.csv`
   - Place it in the `data/raw/` directory

## Usage

### Running the Analysis
1. Start Jupyter Notebook:
```bash
jupyter notebook
```

2. Open and run notebooks in order:
   - `01_data_exploration.ipynb` - Exploratory Data Analysis
   - `02_data_preprocessing.ipynb` - Data cleaning and preprocessing
   - `03_model_development.ipynb` - Model training and selection
   - `04_model_evaluation.ipynb` - Model evaluation and comparison

### Using the Scripts
```python
# Example usage
from src.model_training import train_models
from src.evaluation import evaluate_model

# Train models
models = train_models(X_train, y_train)

# Evaluate best model
results = evaluate_model(best_model, X_test, y_test)
```

## Models Used
- **Decision Tree Classifier/Regressor**
- **Logistic Regression** (for classification tasks)
- **Random Forest Classifier/Regressor**
- **Additional models:** Support Vector Machine, Gradient Boosting

## Evaluation Metrics
- **Regression:** MAE, MSE, RMSE, R²
- **Classification:** Accuracy, Precision, Recall, F1-score, ROC-AUC

## Key Findings
- [To be updated after analysis]
- Most important features for predicting math scores
- Model performance comparison
- Insights about student performance factors

## Results
| Model | RMSE | R² Score | MAE |
|-------|------|----------|-----|
| Random Forest | TBD | TBD | TBD |
| Decision Tree | TBD | TBD | TBD |
| Linear Regression | TBD | TBD | TBD |

## Feature Importance
[To be updated with visualization and ranking of most important features]

## Contributing
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-feature`)
3. Commit your changes (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin feature/new-feature`)
5. Create a Pull Request

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact
- **Author:** [Your Name]
- **Email:** [your.email@example.com]
- **LinkedIn:** [Your LinkedIn Profile]

## Acknowledgments
- Dataset provided by [Kaggle](https://www.kaggle.com/spscientist/students-performance-in-exams)
- Inspiration from educational data mining research
- Thanks to the open-source community for the tools and libraries used