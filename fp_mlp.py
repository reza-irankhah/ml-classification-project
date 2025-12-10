import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow import keras
from tensorflow.keras import layers
from csv import reader
###############################

################################
dataframe = pd.read_csv('dataset.csv')
list_0 = []
list_1 = []
for item in dataframe['target']:
    if item == 0:
        list_0.append(item)
    else:
        list_1.append(item)
print('len class 0: ', len(list_0))
print('len class 1: ', len(list_1))

val_dataframe = dataframe.sample(frac=0.2, random_state=1337)
train_dataframe = dataframe.drop(val_dataframe.index)


def dataframe_to_dataset(dataframe):
    dataframe = dataframe.copy()
    labels = dataframe.pop("target")
    ds = tf.data.Dataset.from_tensor_slices((dict(dataframe), labels))
    ds = ds.shuffle(buffer_size=len(dataframe))
    return ds


train_ds = dataframe_to_dataset(train_dataframe)
val_ds = dataframe_to_dataset(val_dataframe)

train_ds = train_ds.batch(128)
val_ds = val_ds.batch(128)


from keras import backend as K
def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))


def show_plot(loss, val_loss, precision, recal, f_messure):
    import matplotlib.pyplot as plt

    plt.subplot(2, 2, 1)
    plt.grid(True)
    plt.plot(loss); plt.plot(val_loss); plt.xlabel('Epochs'); plt.ylabel('Accuracy'); plt.title('Losses')
    plt.legend(['loss', 'val_loss'])

    plt.subplot(2, 2, 2)
    plt.grid(True)
    plt.plot(precision); plt.xlabel('Epochs'); plt.ylabel('precision'); plt.title('Precision')

    plt.subplot(2, 2, 3)
    plt.grid(True)
    plt.plot(recal); plt.xlabel('Epochs'); plt.ylabel('recall'); plt.title('Recall')

    plt.subplot(2, 2, 4)
    plt.grid(True)
    plt.plot(f_messure); plt.xlabel('Epochs'); plt.ylabel('f'); plt.title('F_messure')

    plt.show()


from tensorflow.keras.layers import IntegerLookup
from tensorflow.keras.layers import Normalization
from tensorflow.keras.layers import StringLookup


def encode_numerical_feature(feature, name, dataset):

    # Create a Normalization layer for our feature
    normalizer = Normalization()

    # Prepare a Dataset that only yields our feature
    feature_ds = dataset.map(lambda x, y: x[name])
    feature_ds = feature_ds.map(lambda x: tf.expand_dims(x, -1))

    # Learn the statistics of the data
    normalizer.adapt(feature_ds)

    # Normalize the input feature
    encoded_feature = normalizer(feature)
    return encoded_feature


def encode_categorical_feature(feature, name, dataset, is_string):
    lookup_class = StringLookup if is_string else IntegerLookup
    # Create a lookup layer which will turn strings into integer indices
    lookup = lookup_class(output_mode="binary")

    # Prepare a Dataset that only yields our feature
    feature_ds = dataset.map(lambda x, y: x[name])
    feature_ds = feature_ds.map(lambda x: tf.expand_dims(x, -1))

    # Learn the set of possible string values and assign them a fixed integer index
    lookup.adapt(feature_ds)

    # Turn the string input into integer indices
    encoded_feature = lookup(feature)
    return encoded_feature

##############################

# Categorical features encoded as integers
workclass = keras.Input(shape=(1,), name="workclass", dtype="int64")
education = keras.Input(shape=(1,), name="education", dtype="int64")
marital_status = keras.Input(shape=(1,), name="marital-status", dtype="int64")
relationship = keras.Input(shape=(1,), name="relationship", dtype="int64")
race = keras.Input(shape=(1,), name="race", dtype="int64")
sex = keras.Input(shape=(1,), name="sex", dtype="int64")
native_country = keras.Input(shape=(1,), name="native-country", dtype="int64")

# Numerical features
age = keras.Input(shape=(1,), name="age")
fnlwgt = keras.Input(shape=(1,), name="fnlwgt")
education_num = keras.Input(shape=(1,), name="education-num")
occupation = keras.Input(shape=(1,), name="occupation")
capital_gain = keras.Input(shape=(1,), name="capital-gain")
capital_loss = keras.Input(shape=(1,), name="capital-loss")
hours_per_week = keras.Input(shape=(1,), name="hours-per-week")

all_inputs = [
    workclass,
    education,
    marital_status,
    relationship,
    race,
    sex,
    native_country,
    age,
    fnlwgt,
    education_num,
    occupation,
    capital_gain,
    capital_loss,
    hours_per_week,
]

# Integer categorical features
workclass_encoded = encode_categorical_feature(workclass, "workclass", train_ds, False)
education_encoded = encode_categorical_feature(education, "education", train_ds, False)
marital_status_encoded = encode_categorical_feature(marital_status, "marital-status", train_ds, False)
relationship_encoded = encode_categorical_feature(relationship, "relationship", train_ds, False)
race_encoded = encode_categorical_feature(race, "race", train_ds, False)
sex_encoded = encode_categorical_feature(sex, "sex", train_ds, False)
native_country_encoded = encode_categorical_feature(native_country, "native-country", train_ds, False)

# Numerical features
age_encoded = encode_numerical_feature(age, "age", train_ds)
fnlwgt_encoded = encode_numerical_feature(fnlwgt, "fnlwgt", train_ds)
education_num_encoded = encode_numerical_feature(education_num, "education-num", train_ds)
occupation_encoded = encode_numerical_feature(occupation, "occupation", train_ds)
capital_gain_encoded = encode_numerical_feature(capital_gain, "capital-gain", train_ds)
capital_loss_encoded = encode_numerical_feature(capital_loss, "capital-loss", train_ds)
hours_per_week_encoded = encode_numerical_feature(hours_per_week, "hours-per-week", train_ds)

all_features = layers.concatenate(
    [
        workclass_encoded,
        education_encoded,
        marital_status_encoded,
        relationship_encoded,
        race_encoded,
        sex_encoded,
        native_country_encoded,
        age_encoded,
        fnlwgt_encoded,
        education_num_encoded,
        occupation_encoded,
        capital_gain_encoded,
        capital_loss_encoded,
        hours_per_week_encoded,
    ]
)
x = layers.Dense(64, activation="relu")(all_features)
# x = layers.Dense(16, activation="relu")(x)

x = layers.Dropout(0.4)(x)
x = layers.Dense(16, activation="relu")(x)
x = layers.Dropout(0.4)(x)

output = layers.Dense(1, activation="sigmoid")(x)
model = keras.Model(all_inputs, output)
# model.summary()

from tensorflow import keras
# model.compile("adam", "binary_crossentropy", metrics=["accuracy", precision_m, recall_m, f1_m])
model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.005), loss="binary_crossentropy", metrics=["accuracy", precision_m, recall_m, f1_m])


# keras.utils.plot_model(model, show_shapes=True, rankdir="LR")

network_history = model.fit(train_ds, epochs=20, validation_data=val_ds)

history = network_history.history

show_plot(loss=history['loss'], val_loss=history['val_loss'], precision=history['precision_m'], recal=history['recall_m'], f_messure=history['f1_m'])
# model.save_weights('model_weights')
######################################################################
def load_file(filename):
    dataset = list()
    with open(filename, 'r') as file:
        csv_reader = reader(file)
        for row in csv_reader:
            if not row:
                continue
            dataset.append(row)
    return dataset


dataset = load_file('dataset_t.csv')
del dataset[0]
n = 1
for list in dataset:
    sample = {
        "age": int(list[0]),
        "workclass": int(list[1]),
        "fnlwgt": int(list[2]),
        "education": int(list[3]),
        "education-num": int(list[4]),
        "marital-status": int(list[5]),
        "occupation": int(list[6]),
        "relationship": int(list[7]),
        "race": int(list[8]),
        "sex": int(list[9]),
        "capital-gain": int(list[10]),
        "capital-loss": int(list[11]),
        "hours-per-week": int(list[12]),
        "native-country": int(list[13]),
    }

    input_dict = {name: tf.convert_to_tensor([value]) for name, value in sample.items()}
    predictions = model.predict(input_dict)
    print("class number", str(n), "percentage of being rich", format(100 * predictions[0][0], '.1f'))
    n += 1
