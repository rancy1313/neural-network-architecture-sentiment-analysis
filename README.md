# neural-network-architecture-sentiment-analysis
Comparison of sequential vs. functional neural network architectures for sentiment analysis of reviews, using BiLSTM layers and specialized activation functions

## Project Overview
This project compares sequential and functional neural network architectures to determine which more accurately and confidently predicts sentiment in review data. Using the UCI Sentiment Labeled Sentences dataset containing reviews from Amazon, IMDb, and Yelp, the analysis implements and evaluates two hybrid recurrent and feedforward neural network models of comparable complexity but different architectures.

## Research Question
Which neural network architecture, sequential or functional, more accurately and confidently predicts sentiment in review data to help organizations better understand user sentiment?

## Methodology
The analysis follows a comprehensive approach to compare the neural network architectures:

1. **Data Exploration and Preparation**:
   - Combined reviews from multiple sources (Amazon, IMDb, Yelp)
   - Identified and removed unusual characters
   - Analyzed vocabulary size and sequence length distribution
   - Determined optimal maximum sequence length (31 words at 95th percentile)

2. **Text Processing**:
   - Tokenized text using spaCy's en_core_web_md model
   - Converted reviews to word embedding sequences (300-dimensional vectors)
   - Handled out-of-vocabulary words
   - Applied sequence padding for uniform length

3. **Model Architecture**:
   - **Functional Model**: Implemented a dual-branch architecture with specialized activation functions
     - One branch using standard ReLU activation (focusing on positive values)
     - One branch using inverse ReLU activation (focusing on negative values)
     - Both branches merged before final classification
   
   - **Sequential Model**: Implemented a traditional sequential architecture with comparable complexity
     - Using standard ReLU activation functions
     - Fully connected layers throughout

4. **Model Training and Evaluation**:
   - Applied identical hyperparameters to both models
   - Used binary cross-entropy loss function and Adam optimizer
   - Implemented regularization techniques (dropout, L2 regularization)
   - Evaluated performance using accuracy and loss metrics

## Key Findings
- The functional model consistently outperformed the sequential model:
  - Functional model: 70.92% test accuracy, 0.6851 test loss
  - Sequential model: 66.47% test accuracy, 1.0873 test loss
  
- The dual-branch functional architecture showed better generalization to unseen data with a smaller gap between training and validation metrics

- Both models showed signs of overfitting, but the effect was more pronounced in the sequential model despite identical regularization strategies

- Training visualizations suggest both models reached peak validation performance around 15 epochs

## Technologies Used
- Python
- TensorFlow and Keras
- spaCy (en_core_web_md model)
- NumPy
- Regular expressions (re)
- Bidirectional LSTM neural networks

## Future Work
- Reduce early stopping patience from 5 to 2 epochs to minimize overfitting
- Limit total training epochs to 15 to prevent performance degradation
- Explore if branches in the functional model learned complementary patterns related to sentiment
- Test performance differences across review domains (Amazon vs. IMDb vs. Yelp)
- Increase model complexity with additional nodes, layers, and activation functions
- Collect more training data to improve overall model performance

*This project was completed as part of a Data Analytics course (D213).*
