#relevant modules
import pandas as pd 
import tensorflow as tf 
from matplotlib import pyplot as plt 

#adjust granularity of reporting
pd.options.display.max_rows=10
pd.options.display.float_format="{:.1f}".format

#importing the dataset
training_df = pd.read_csv(filepath_or_buffer="california_housing_train.csv")

#scaling the label
training_df["median_house_value"] /= 1000.0

print("\nfirst row of the pandas Dataframe:\n")
training_df.head()

print("\nStats of data:\n")
training_df.describe()

# Anomaly: The maximum value (max) of several columns seems very
# high compared to the other quantiles. For example,
# example the total_rooms column. Given the quantile
# values (25%, 50%, and 75%), you might expect the 
# max value of total_rooms to be approximately 
# 5,000 or possibly 10,000. However, the max value 
# is actually 37,937.

#build and train a model
def build_model(my_learning_rate):
    """Create and compile a simple linear regression model."""
    # Most simple tf.keras models are sequential.
    model = tf.keras.models.Sequential()

    #topography: single node in single layer
    model.add(tf.keras.layers.Dense(units=1, input_shape=(1,)))

    model.compile(optimizer=tf.keras.optimizers.experimental.RMSprop(learning_rate=my_learning_rate),
                 loss = "mean_squared_error",
                 metrics=[tf.keras.metrics.RootMeanSquaredError()] )
    return model


def train_model(model, df, feature, label, epochs, batch_size):
    """Train the model by feeding it data."""

    #feature and label in model for it to train for specific epochs
    history = model.fit(x=df[feature],
                        y=df[label],
                        batch_size=batch_size,
                        epochs=epochs)
    
    #Gather trained model's weight and bias
    trained_weight = model.get_weights()[0]
    trained_bias = model.get_weights()[1]

    #store epoch list
    epochs=history.epoch

    #error for each epoch
    hist=pd.DataFrame(history.history)

    #rms error of each epoch
    rmse=hist["root_mean_squared_error"]

    return trained_weight, trained_bias, epochs, rmse


def plot_the_model(trained_weight, trained_bias, feature, label):
    """Plot the trained model against 200 random training examples"""

    #axis label
    plt.xlabel(feature)
    plt.ylabel(label)

    #scatter plot from 200 random pointers of the dataset
    random_examples = training_df.sample(n=200)
    plt.scatter(random_examples[feature], random_examples[label])

    #red line for the model
    x0 = 0
    y0 = trained_bias
    x1 = random_examples[feature].max()
    y1 = trained_bias + (trained_weight * x1)
    plt.plot([x0, x1], [y0, y1], color="red")

    plt.show()

def plot_the_loss_curve(epochs, rmse):
    """Plot a curve of loss vs epochs"""

    plt.figure()
    plt.xlabel("Epoch")
    plt.ylabel("Root Mean Squared Error")

    plt.plot(epochs, rmse, label="Loss")
    plt.legend()
    plt.ylim([rmse.min()*0.97, rmse.max()])
    plt.show()

def predict_house_values(n, feature, label):
  """Predict house values based on a feature."""

  batch = training_df[feature][10000:10000 + n]
  predicted_values = my_model.predict_on_batch(x=batch)

  print("feature   label          predicted")
  print("  value   value          value")
  print("          in thousand$   in thousand$")
  print("--------------------------------------")
  for i in range(n):
    print ("%5.0f %6.0f %15.0f" % (training_df[feature][10000 + i],
                                   training_df[label][10000 + i],
                                   predicted_values[i][0] ))

#@title Double-click to view a possible solution to Task 4.

# Define a synthetic feature
training_df["rooms_per_person"] = training_df["total_rooms"] / training_df["population"]
my_feature = "rooms_per_person"

# Tune the hyperparameters.
learning_rate = 0.06
epochs = 24
batch_size = 30

# Don't change anything below this line.
my_model = build_model(learning_rate)
weight, bias, epochs, mae = train_model(my_model, training_df,
                                        my_feature, my_label,
                                        epochs, batch_size)

plot_the_loss_curve(epochs, mae)
predict_house_values(15, my_feature, my_label)


# Generate a correlation matrix.
training_df.corr()
