import tensorflow as tf

data_tes = tf.keras.preprocessing.image_dataset_from_directory(
    'dataset/data_tes/',
    image_size=(224, 224),
    shuffle=False,
    batch_size=32)


