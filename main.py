#relevant modules
import pandas as pd 
import tensorflow as tf 
from matplotlib import pyplot as plt 

#adjust granularity of reporting
pd.options.display.max_rows=10
pd.options.display.float_format="{:.1f}".format

#importing the dataset
train_df = pd.read_csv("https://download.mlcc.google.com/mledu-datasets/california_housing_train.csv")
test_df = pd.read_csv("https://download.mlcc.google.com/mledu-datasets/california_housing_test.csv")

scale_factor = 1000.0

#scaling the label
train_df["median_house_value"] /= scale_factor
test_df["median_house_value"] /= scale_factor

print("\nfirst row of the pandas Dataframe:\n")
train_df.head()

print("\nStats of data:\n")
train_df.describe()

# Anomaly: The maximum value (max) of several columns seems very
# high compared to the other quantiles. For example,
# example the total_rooms column. Given the quantile
# values (25%, 50%, and 75%), you might expect the 
# max value of total_rooms to be approximately 
# 5,000 or possibly 10,000. However, the max value 
# is actually 37,937.

#build and train a model
#@title Define the functions that build and train a model
def build_model(my_learning_rate):
  """Create and compile a simple linear regression model."""
  # Most simple tf.keras models are sequential.
  model = tf.keras.models.Sequential()

  # Add one linear layer to the model to yield a simple linear regressor.
  model.add(tf.keras.layers.Dense(units=1, input_shape=(1,)))

  # Compile the model topography into code that TensorFlow can efficiently
  # execute. Configure train to minimize the model's mean squared error. 
  model.compile(optimizer=tf.keras.optimizers.experimental.RMSprop(learning_rate=my_learning_rate),
                loss="mean_squared_error",
                metrics=[tf.keras.metrics.RootMeanSquaredError()])

  return model               


def train_model(model, df, feature, label, my_epochs, 
                my_batch_size=None, my_validation_split=0.1):
  """Feed a dataset into the model in order to train it."""

  history = model.fit(x=df[feature],
                      y=df[label],
                      batch_size=my_batch_size,
                      epochs=my_epochs,
                      validation_split=my_validation_split)

  # Gather the model's trained weight and bias.
  trained_weight = model.get_weights()[0]
  trained_bias = model.get_weights()[1]

  # The list of epochs is stored separately from the 
  # rest of history.
  epochs = history.epoch
  
  # Isolate the root mean squared error for each epoch.
  hist = pd.DataFrame(history.history)
  rmse = hist["root_mean_squared_error"]

  return epochs, rmse, history.history   

print("Defined the build_model and train_model functions.")

def plot_the_model(trained_weight, trained_bias, feature, label):
    """Plot the trained model against 200 random train examples"""

    #axis label
    plt.xlabel(feature)
    plt.ylabel(label)

    #scatter plot from 200 random pointers of the dataset
    random_examples = train_df.sample(n=200)
    plt.scatter(random_examples[feature], random_examples[label])

    #red line for the model
    x0 = 0
    y0 = trained_bias
    x1 = random_examples[feature].max()
    y1 = trained_bias + (trained_weight * x1)
    plt.plot([x0, x1], [y0, y1], color="red")

    plt.show()

#@title Define the plotting function

def plot_the_loss_curve(epochs, mae_train, mae_validation):
  """Plot a curve of loss vs. epoch."""

  plt.figure()
  plt.xlabel("Epoch")
  plt.ylabel("Root Mean Squared Error")

  plt.plot(epochs[1:], mae_train[1:], label="train Loss")
  plt.plot(epochs[1:], mae_validation[1:], label="Validation Loss")
  plt.legend()
  
  # We're not going to plot the first epoch, since the loss on the first epoch
  # is often substantially greater than the loss for other epochs.
  merged_mae_lists = mae_train[1:] + mae_validation[1:]
  highest_loss = max(merged_mae_lists)
  lowest_loss = min(merged_mae_lists)
  delta = highest_loss - lowest_loss
  print(delta)

  top_of_y_axis = highest_loss + (delta * 0.05)
  bottom_of_y_axis = lowest_loss - (delta * 0.05)
   
  plt.ylim([bottom_of_y_axis, top_of_y_axis])
  plt.show()  

print("Defined the plot_the_loss_curve function.")

def predict_house_values(n, feature, label):
  """Predict house values based on a feature."""

  batch = train_df[feature][10000:10000 + n]
  predicted_values = my_model.predict_on_batch(x=batch)

  print("feature   label          predicted")
  print("  value   value          value")
  print("          in thousand$   in thousand$")
  print("--------------------------------------")
  for i in range(n):
    print ("%5.0f %6.0f %15.0f" % (train_df[feature][10000 + i],
                                   train_df[label][10000 + i],
                                   predicted_values[i][0] ))


#hyperparameter:
learning_rate = 0.08
epochs = 70
batch_size = 100

# Split the original train set into a reduced train set and a
# validation set. 
validation_split = 0.1 #closer when valsplit<0.15

my_feature = "median_income" #median income of specific city block
my_label = "median_house_value" #on a specific city block

shuffled_train_df = train_df.reindex(np.random.permutation(train_df.index))


my_model = build_model(learning_rate)
epochs, rmse, history = train_model(my_model, shuffled_train_df, my_feature, my_label, epochs, batch_size, validation_split)

plot_the_loss_curve(epochs, history["root_mean_squared_error"],
                    history["val_root_mean_squared_error"])

x_test = test_df[my_feature]
y_test = test_df[my_label]
result = my_model.evaluate(x_test, y_test, batch_size=batch_size)

#rmse value were similar enough