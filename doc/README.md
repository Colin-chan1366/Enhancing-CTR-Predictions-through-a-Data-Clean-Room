We have finished all 3 parts in task2.

For the first part, we aggregated and compared important features including age, location, type between users who clicked and all users.

For the second part, we developed a random forest model after generating dummy variables with high difference between users who clicked and all users.

For the last part, we used a GAN model to generate synthesized data.

Author: Colin & Bike

###### GAN.ipynb Implementation Guidelines:
- Genarative model for synthetic data generation in ads.csv:

Data merging: The code analyzes the common columns of the two datasets and tries to find a way to merge the datasets. However, relying solely on the existing common columns ("label", "u_newsCatInterestsST", "u_refreshTimes", "u_feedLifeCycle") may not be able to effectively merge user-level data.

Data cleaning and feature engineering: The code processes the object type column in the ads dataset, splits the string type data into lists, and unifies the list length. These steps are to convert the raw data into numerical features that can be processed by the machine learning model.

- A simple Generative Adversarial Network (GAN) composed of a Generator and a Discriminator.

**Generator:**

*   Takes an input vector of size `input_size` (likely representing random noise).
*   Passes it through a series of fully connected layers (`nn.Linear`) with ReLU activation functions (`nn.ReLU`) to gradually increase the dimensionality and learn complex representations.
*   The final layer outputs a vector of size `output_size`, representing the generated data (e.g., an image).

**Discriminator:**

*   Takes an input vector of size `input_size` (which could be either real data or generated data).
*   Uses fully connected layers with LeakyReLU activations (`nn.LeakyReLU`) to learn features and gradually reduce dimensionality.
*   The final layer outputs a single scalar value between 0 and 1 (`nn.Sigmoid`), representing the probability of the input being real data (close to 1) or fake (close to 0).
