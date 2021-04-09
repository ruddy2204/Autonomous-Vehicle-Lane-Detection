import argparse
import fnmatch
import json
import os
import random
import nvidia_model
import numpy as np
from PIL import Image
from tensorflow import keras

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    do_plot = True
except ImportError:
    do_plot = False


def parse_img_filepath(filepath):
    basename = os.path.basename(filepath)
    f = basename[:-4]
    f = f.split("_")

    steering = float(f[3])
    throttle = float(f[5])

    data = {"steering": steering, "throttle": throttle}

    return data

def load_json(filename):
    with open(filename, "rt") as fp:
        data = json.load(fp)
    return data


def generator(samples, batch_size=32, perc_to_augment=0.5):
    num_samples = len(samples)

    while 1:
        random.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset: offset + batch_size]

            images = []
            controls = []

            for fullpath in batch_samples:
                try:
                    frame_number = os.path.basename(fullpath).split("_")[0]
                    json_filename = os.path.join(os.path.dirname(fullpath), "record_" + frame_number + ".json")
                    data = load_json(json_filename)
                    steering = float(data["user/angle"])
                    throttle = float(data["user/throttle"]) / 1.0

                    try:
                        image = Image.open(fullpath)
                    except:
                        image = None

                    if image is None:
                        print("failed to open", fullpath)
                        continue

                    image = np.array(image, dtype=np.float32)
                    images.append(image)
                    controls.append([steering, throttle])


                except Exception as e:
                    print(e)
                    print("throw an exception on:", fullpath)
                    yield [], []

                X_train = np.array(images)
                y_train = np.array(controls)
                yield X_train, y_train


def get_files(filemask):
    filemask = os.path.expanduser(filemask)
    path, mask = os.path.split(filemask)

    matches = []
    for root, dirnames, filenames in os.walk(path):
        for filename in fnmatch.filter(filenames, mask):
            matches.append(os.path.join(root, filename))
    return matches

def train_test_split(lines, test_perc):
    train = []
    test = []

    for line in lines:
        if random.uniform(0.0, 1.0) < test_perc:
            test.append(line)
        else:
            train.append(line)

    return train, test

def make_generators(inputs, limit=None, batch_size=32, aug_perc=0.0):
    lines = get_files(inputs)

    if limit is not None:
        lines = lines[:limit]
        print("limiting to %d files" % len(lines))

    train_samples, validation_samples = train_test_split(lines, test_perc=0.2)

    train_generator = generator(train_samples, batch_size=batch_size)
    validation_generator = generator(validation_samples, batch_size=batch_size)

    n_train = len(train_samples)
    n_val = len(validation_samples)

    return train_generator, validation_generator, n_train, n_val


def train_model(model_name, epochs=50, inputs="./log/*.jpg", limit=None):
    model = nvidia_model.get_nvidia_model(2)

    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=6, verbose=0),
        keras.callbacks.ModelCheckpoint(model_name, monitor="val_loss", save_best_only=True, verbose=0),
    ]

    batch_size = 128
    train_generator, validation_generator, n_train, n_val = make_generators(inputs, limit=limit, batch_size=batch_size)

    steps_per_epoch = n_train // batch_size
    validation_steps = n_val // batch_size

    print("steps_per_epoch", steps_per_epoch, "validation_steps", validation_steps)

    history = model.fit_generator(
        train_generator,
        steps_per_epoch=steps_per_epoch,
        validation_data=validation_generator,
        validation_steps=validation_steps,
        epochs=epochs,
        verbose=1,
        callbacks=callbacks,
    )

    try:
        if do_plot:
            plt.plot(history.history["loss"])
            plt.plot(history.history["val_loss"])
            plt.title("model loss")
            plt.ylabel("loss")
            plt.xlabel("epoch")
            plt.legend(["train", "test"], loc="upper left")
            plt.savefig(model_name + "loss.png")
            plt.show()
    except:
        print("problems with loss graph")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="train script")
    parser.add_argument("--model", type=str, help="model name")
    parser.add_argument("--epochs", type=int, default=100, help="number of epochs")
    parser.add_argument("--inputs", default="./log/*.jpg", help="input mask to gather images")
    parser.add_argument("--limit", type=int, default=None, help="max number of images to train with")
    args = parser.parse_args()

    train_model(args.model, epochs=args.epochs, limit=args.limit, inputs=args.inputs)