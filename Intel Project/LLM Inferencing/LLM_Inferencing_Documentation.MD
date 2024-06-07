# Step 3: Simple LLM Inference on CPU

This step involves performing inference using a pre-trained Large Language Model (LLM) on a CPU. We will cover setting up the environment, loading a model, running inference, and understanding the results.

## 1. Launch Jupyter Notebook on IDC

1. **Access Jupyter Notebook**:
   - Log in to the [Intel Developer Cloud](https://www.intel.com/content/www/us/en/developer/tools/devcloud/services.html).
   - Navigate to the "Notebooks" section.
   - Click "Launch Notebook" to start a new Jupyter Notebook session.

## 2. Install Necessary Libraries

1. **Install Libraries**:
   - Ensure the required libraries (`transformers`, `torch`, `scikit-learn`) are installed.
   - Open a new cell in the Jupyter Notebook and run the necessary installation commands.

## 3. Load Pre-trained Model and Tokenizer

1. **Import Libraries**:
   - Import the necessary libraries from the `transformers` package.

2. **Set Device to CPU**:
   - Ensure that the model runs on the CPU. If a GPU is available, you can explicitly specify the device as CPU.

3. **Load Pre-trained Model**:
   - Use the `pipeline` function to load a pre-trained model and tokenizer. In this example, we use the BERT model for the fill-mask task.

## 4. Perform Inference on CPU

1. **Perform Inference**:
   - Provide input text to the model and get the predicted output. The fill-mask task predicts the masked word in the sentence.

2. **Understand the Output**:
   - The model returns the predicted token and a confidence score for each prediction. Analyze the top predictions to see how well the model performs.

## 5. Experiment with Different Inputs

1. **Try Various Inputs**:
   - Test the model with different sentences to observe its behavior in different contexts.

2. **Analyze Model Behavior**:
   - Observe how the model fills in the masked tokens based on different contexts. Note how the model's predictions vary with different inputs.

## 6. Detailed Example: Understanding Results

1. **Example Inference**:
   - Consider the sentence "Artificial intelligence is a [MASK] field." The model might predict tokens like "rapidly," "growing," "complex," etc., with varying confidence scores.

2. **Interpreting Scores**:
   - The `Score` represents the model's confidence in its prediction. Higher scores indicate higher confidence.

## 7. Document the Process

1. **Record Steps and Observations**:
   - Document each step you performed and the results obtained. Include the code, input sentences, and output predictions.

2. **Save Notebook**:
   - Save your Jupyter Notebook with all the code, inputs, and outputs for future reference and reporting. This documentation will be useful when creating the final project report.

## 8. Tips and Best Practices

1. **Keep Code Organized**:
   - Use comments and markdown cells to explain each step and its purpose. This helps in understanding the workflow and makes it easier to review later.

2. **Experiment and Iterate**:
   - Try different models and tasks available in the `transformers` library to gain a deeper understanding of their capabilities. Experiment with various input sentences to see how the model handles different contexts.

3. **Review Documentation**:
   - Regularly refer to the [Hugging Face Transformers documentation](https://huggingface.co/transformers/) for detailed information on models, pipelines, and usage.

By following these detailed steps, you will perform simple LLM inference on a CPU, gaining practical experience with pre-trained models and understanding how to use them for text prediction tasks. This foundational knowledge is essential for more advanced tasks like fine-tuning LLM models.
