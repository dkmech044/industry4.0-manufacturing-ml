[![Run in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dkmech044/industry4.0-manufacturing-ml/blob/main/weld_quality_extended_logistic.ipynb)


# Weld Quality Prediction Using Machine Learning (Extended)

In this project, I developed a progressively complex machine learning pipeline to predict weld quality based on high-strain-rate impact parameters such as impact velocity, flyer angle, yield strength, and waveform type.
Initially, I implemented Logistic Regression to establish a baseline for binary weld classification (Good vs Defective), achieving around 45% accuracy. Recognizing the limitations of linear models for complex manufacturing processes, I next adopted a Random Forest Classifier, improving prediction accuracy to 60% by capturing non-linear feature interactions.

To further enhance model performance, I built a Feedforward Neural Network (FNN) with two hidden layers, achieving a moderate 60% accuracy. Finally, I designed an Advanced Neural Network architecture with three hidden layers and dropout regularization, which pushed the model's performance to an accuracy of 65% and a recall of 83% for detecting defective welds — critical for ensuring manufacturing quality and safety.

This work demonstrates a full-cycle Industry 4.0 approach, combining feature engineering, model selection, hyperparameter tuning, and deep learning for smart manufacturing applications. It also reflects my ability to bridge mechanical engineering fundamentals with modern AI techniques, preparing me to contribute meaningfully to research and teaching in next-generation manufacturing systems.

