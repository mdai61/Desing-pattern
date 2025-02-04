Predicting H1 Target for November 2024: A Predictive Modeling Project with OOP & Design Patterns
Overview
This project extends a previous predictive modeling implementation by incorporating object-oriented programming (OOP) principles and software design patterns. It enhances code maintainability, reusability, and scalability while maintaining the original goal of forecasting target categories based on historical OHLCV data resampled into the H1 (1-hour) timeframe.
Key improvements:
•	Refactored Codebase: Uses OOP principles (encapsulation, inheritance, polymorphism, abstraction).
•	Design Patterns: Implemented Template Method, Strategy, and Decorator patterns.
•	Graphical User Interface (GUI): Integrated Kivy-based UI for model selection and execution.
•	Automated Testing: Developed unit tests for trainers and visualizers.
•	WHL Packaging: Created a distributable Python package (.whl) for easy installation.
________________________________________
Target Categories
The target categories are based on the percentage change in the Close price over the next 10 H1 candles:
•	A: (< -5%)
•	B: (-5% to -2%)
•	C: (-2% to +2%)
•	D: (+2% to +5%)
•	E: (> +5%)
Additionally, a momentum target based on the Relative Strength Index (RSI) was implemented.
________________________________________
Dataset
The dataset consists of historical M1 (1-minute) OHLCV data, resampled into H1 (1-hour) data. It includes:
•	OHLCV Features: Open, High, Low, Close, Volume aggregated over 1-hour intervals.
•	Engineered Features: Lagged values, moving averages, RSI-based momentum metrics.
________________________________________
Key Improvements in This Project
1. Object-Oriented Programming (OOP) Principles
•	Encapsulation: Modularized code into separate classes (DataProcessor, ModelTrainer, FeatureSelector, Visualizer).
•	Inheritance: Shared functionality implemented in abstract base classes (BaseTrainer, BaseVisualizer), extended by concrete classes (XGBoostTrainer, XGBoostVisualizer).
•	Polymorphism: Training and visualization are dynamically handled through abstract base classes, enabling easy extensibility.
•	Abstraction: High-level classes define workflows while leaving implementation details to subclasses.
2. Design Patterns Used
•	Template Method Pattern: 
o	BaseTrainer defines a high-level workflow for model training (preprocessing → training → evaluation → logging).
o	Concrete implementations (XGBoostTrainer) handle specific implementations.
•	Strategy Pattern: 
o	Training strategies are encapsulated in separate trainer classes, allowing easy swapping or extension.
•	Decorator Pattern: 
o	The log_results decorator dynamically adds logging functionality to evaluation methods without modifying their core logic.
3. GUI Integration with Kivy
A Kivy-based graphical user interface (GUI) was developed to allow users to:
•	Select start and end dates for dataset filtering.
•	Choose a test period for model evaluation.
•	Run the model and visualize results dynamically (confusion matrix and performance metrics).
4. Automated Testing
•	Unit tests were developed for the XGBoostTrainer and XGBoostVisualizer classes.
•	Tests ensure data preprocessing, training, evaluation, and visualization work correctly.
5. WHL Packaging for Easy Distribution
•	The project was packaged into a .whl (Wheel) file using setuptools for easy distribution and installation.
________________________________________
Project Structure
project_name/
│
├── data/                     # Data files (if applicable)
├── scripts/                  # Core scripts
│   ├── __init__.py           # Package initializer
│   ├── data_processor.py     # Data processing logic
│   ├── visualizer.py         # Visualization functions
│   ├── feature_selector.py   # Feature selection
│   ├── xgboost_trainer.py    # Model training logic
│
├── ui/                       # GUI-related components
│   ├── __init__.py           # Package initializer
│   ├── kivy_app.py           # Kivy-based user interface
│
├── tests/                    # Unit tests
├── dist/                     # WHL package distribution
├── setup.py                  # Packaging configuration
├── requirements.txt          # Dependencies
├── README.md                 # Project documentation
└── main.py                   # Main execution file
________________________________________
How to Run the Project
1. Clone the Repository
git clone https://github.com/yourusername/project_name.git
cd project_name
2. Install Dependencies
pip install -r requirements.txt
3. Run the Project
Option 1: Run in CLI Mode
python main.py
Option 2: Run the GUI
python main.py gui
Or directly:
python ui/kivy_app.py
4. Run Tests
pytest tests/
5. Install as a Python Package (WHL)
If you received a .whl file, install it with:
pip install dist/project_name-0.1.0-py3-none-any.whl
________________________________________
Results and Evaluation
•	Confusion Matrix: Evaluated model performance on Target 1 predictions for November 2024.
•	Best Model: XGBoost showed the best accuracy and interpretability.
•	Visualization: Time-series plots and confusion matrices enhance interpretability.
________________________________________
Tools and Libraries Used
Programming Language
•	Python
Key Libraries
Library	Purpose
Pandas	Data processing and resampling
NumPy	Numerical computations
Seaborn & Matplotlib	Visualization
Scikit-learn	Feature selection & evaluation metrics
XGBoost	Model training and prediction
Kivy	GUI development
Setuptools & Wheel	WHL package distribution
________________________________________
Key Differences from the Previous Project
Aspect	Previous Implementation	Current Implementation
Code Structure	Monolithic	Modular OOP-based design
Design Patterns	None	Template, Strategy, Decorator
Scalability	Limited	Easily extendable (new models, visualizations)
GUI	None	Kivy-based interactive GUI
Testing	Minimal	Unit tests for training & visualization
Distribution	Script-based execution	WHL package for easy installation
________________________________________
Conclusion
This project improves upon a previous predictive modeling approach by incorporating OOP best practices, design patterns, testing, and GUI integration. The resulting scalable, reusable, and maintainable framework allows for easy future enhancements and modifications.
Would you like additional features, such as hyperparameter tuning or a web-based interface? 🚀

