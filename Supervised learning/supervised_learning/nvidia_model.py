from tensorflow.keras.layers import Conv2D, Cropping2D, Dense, Dropout, Flatten, Input, Lambda
from tensorflow.keras.models import Model

def get_nvidia_model(num_outputs):
    row = 120 # Image Width
    col = 160 # Image Height
    ch = 3    # Image Depth - RGB
    outputs = []

    drop = 0.2

    img_in = Input(shape=(row, col, ch), name="img_in")
    x = img_in

    x = Cropping2D(cropping=((10, 0), (0, 0)))(x)  # trim 10 pixels off top
    x = Lambda(lambda x: x / 255.0)(x)  # normalize
    x = Conv2D(24, (5, 5), strides=(2, 2), activation="relu", name="conv2d_1")(x)
    x = Dropout(drop)(x)
    x = Conv2D(32, (5, 5), strides=(2, 2), activation="relu", name="conv2d_2")(x)
    x = Dropout(drop)(x)
    x = Conv2D(64, (5, 5), strides=(2, 2), activation="relu", name="conv2d_3")(x)
    x = Dropout(drop)(x)
    x = Conv2D(64, (3, 3), strides=(1, 1), activation="relu", name="conv2d_4")(x)
    x = Dropout(drop)(x)
    x = Conv2D(64, (3, 3), strides=(1, 1), activation="relu", name="conv2d_5")(x)
    x = Dropout(drop)(x)

    x = Flatten(name="flattened")(x)
    x = Dense(100, activation="relu")(x)
    x = Dropout(drop)(x)
    x = Dense(50, activation="relu")(x)
    x = Dropout(drop)(x)

    outputs.append(Dense(num_outputs, activation="linear", name="steering_throttle")(x))

    model = Model(inputs=[img_in], outputs=outputs)
    model.compile(optimizer="adam", loss="mse", metrics=["acc"])
    return model