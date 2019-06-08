from keras.datasets import mnist
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.np_utils import to_categorical
import numpy as np
from imgaug import augmenters as iaa

class MNIST:
    x_train, y_train, x_test, y_test = None, None, None, None
    train_size, test_size = 0, 0
 
    def __init__(self):
        (self.x_train, self.y_train), (self.x_test, self.y_test) = mnist.load_data()
        # reshape
        self.x_train = self.x_train.reshape(self.x_train.shape[0], 28, 28, 1)
        self.x_test = self.x_test.reshape(self.x_test.shape[0], 28, 28, 1)
        # convert from int to float
        self.x_train = self.x_train.astype('float32')
        self.x_test = self.x_test.astype('float32')
        # rescale values
        self.x_train /= 255.0
        self.x_test /= 255.0
        # Save dataset sizes
        self.train_size = self.x_train.shape[0]
        self.test_size = self.x_test.shape[0]
        # Create one hot array
        self.y_train = to_categorical(self.y_train, 10)
        self.y_test = to_categorical(self.y_test, 10)

    def data_augmentation(self, augment_size=5000): 
        image_generator = ImageDataGenerator(
            rotation_range=10,
            zoom_range = 0.05, 
            width_shift_range=0.05,
            height_shift_range=0.05,
            horizontal_flip=False,
            vertical_flip=False, 
            data_format="channels_last",
            zca_whitening=True)
        # fit data for zca whitening
        image_generator.fit(self.x_train, augment=True)
        # get transformed images
        randidx = np.random.randint(self.train_size, size=augment_size)
        x_augmented = self.x_train[randidx].copy()
        y_augmented = self.y_train[randidx].copy()
        x_augmented = image_generator.flow(x_augmented, np.zeros(augment_size),
                                    batch_size=augment_size, shuffle=False).next()[0]
        # append augmented data to trainset
        self.x_train = np.concatenate((self.x_train, x_augmented))
        self.y_train = np.concatenate((self.y_train, y_augmented))
        self.train_size = self.x_train.shape[0]
        self.test_size = self.x_test.shape[0]

    def next_train_batch(self, batch_size):
        randidx = np.random.randint(self.train_size, size=batch_size)
        epoch_x = self.x_train[randidx]
        epoch_y = self.y_train[randidx]
        return epoch_x, epoch_y
    
    def next_test_batch(self, batch_size):
        randidx = np.random.randint(self.test_size, size=batch_size)
        epoch_x = self.x_test[randidx]
        epoch_y = self.y_test[randidx]
        return epoch_x, epoch_y

    def get_test_images(self):
        return self.x_test

    def get_test_labels(self):
        return self.y_test

    def shuffle_train(self):
        indices = np.random.permutation(self.train_size)
        self.x_train = self.x_train[indices]
        self.y_train = self.y_train[indices]

from keras.datasets import fashion_mnist

class FMNIST:
    x_train, y_train, x_test, y_test = None, None, None, None
    train_size, test_size = 0, 0
 
    def __init__(self):
        (self.x_train, self.y_train), (self.x_test, self.y_test) = fashion_mnist.load_data()
        # reshape
        self.x_train = self.x_train.reshape(self.x_train.shape[0], 28, 28, 1)
        self.x_test = self.x_test.reshape(self.x_test.shape[0], 28, 28, 1)
        # convert from int to float
        self.x_train = self.x_train.astype('float32')
        self.x_test = self.x_test.astype('float32')
        # rescale values
        self.x_train /= 255.0
        self.x_test /= 255.0
        # Save dataset sizes
        self.train_size = self.x_train.shape[0]
        self.test_size = self.x_test.shape[0]
        # Create one hot array
        self.y_train = to_categorical(self.y_train, 10)
        self.y_test = to_categorical(self.y_test, 10)

    def data_augmentation(self, augment_size=5000): 
        self.image_generator = ImageDataGenerator(
            rotation_range=0,
            zoom_range = 0.0, 
            width_shift_range=0.05,
            height_shift_range=0.05,
            horizontal_flip=False,
            vertical_flip=False, 
            data_format="channels_last",
            zca_whitening=False)
        # fit data for zca whitening
        self.image_generator.fit(self.x_train)

    def next_train_batch(self, batch_size, augment=True, p=0.5):
        randidx = np.random.randint(self.train_size, size=batch_size)
        epoch_x = self.x_train[randidx].copy()
        epoch_y = self.y_train[randidx].copy()
        if augment and np.random.uniform()>p:
            epoch_x, epoch_y = self.image_generator.flow(epoch_x, epoch_y, batch_size=batch_size, 
                # save_to_dir="./aug_imgs",
                # save_prefix="IMG",
                ).next()
        return epoch_x, epoch_y
    
    def next_test_batch(self, batch_size):
        randidx = np.random.randint(self.test_size, size=batch_size)
        epoch_x = self.x_test[randidx]
        epoch_y = self.y_test[randidx]
        return epoch_x, epoch_y

    def get_test_images(self):
        return self.x_test

    def get_test_labels(self):
        return self.y_test

    def shuffle_train(self):
        indices = np.random.permutation(self.train_size)
        self.x_train = self.x_train[indices]
        self.y_train = self.y_train[indices]

class FMNIST_aug:
    x_train, y_train, x_test, y_test = None, None, None, None
    train_size, test_size = 0, 0
 
    def __init__(self):
        (self.x_train, self.y_train), (self.x_test, self.y_test) = fashion_mnist.load_data()
        # reshape
        self.x_train = self.x_train.reshape(self.x_train.shape[0], 28, 28, 1)
        self.x_test = self.x_test.reshape(self.x_test.shape[0], 28, 28, 1)
        # convert from int to float
        self.x_train = self.x_train.astype('float32')
        self.x_test = self.x_test.astype('float32')
        # rescale values
        self.x_train /= 255.0
        self.x_test /= 255.0
        # Save dataset sizes
        self.train_size = self.x_train.shape[0]
        self.test_size = self.x_test.shape[0]
        # Create one hot array
        self.y_train = to_categorical(self.y_train, 10)
        self.y_test = to_categorical(self.y_test, 10)

    def data_augmentation(self): 
        self.image_generator = iaa.Sequential([iaa.Fliplr(0.5), # horizontally flip 50% of the images
        # iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*1.0), per_channel=0.5) # Add to 50%
        ])

    def next_train_batch(self, batch_size, augment=True):
        randidx = np.random.randint(self.train_size, size=batch_size)
        epoch_x = self.x_train[randidx].copy()
        epoch_y = self.y_train[randidx].copy()
        if augment:
            epoch_x = self.image_generator.augment_images(epoch_x)
        return epoch_x, epoch_y
    
    def next_test_batch(self, batch_size):
        randidx = np.random.randint(self.test_size, size=batch_size)
        epoch_x = self.x_test[randidx]
        epoch_y = self.y_test[randidx]
        return epoch_x, epoch_y

    def get_test_images(self):
        return self.x_test

    def get_test_labels(self):
        return self.y_test

    def shuffle_train(self):
        indices = np.random.permutation(self.train_size)
        self.x_train = self.x_train[indices]
        self.y_train = self.y_train[indices]

from keras.datasets import cifar10

class CIFAR10:
    x_train, y_train, x_test, y_test = None, None, None, None
    train_size, test_size = 0, 0
 
    def __init__(self):
        (self.x_train, self.y_train), (self.x_test, self.y_test) = cifar10.load_data()
        # reshape
        self.x_train = self.x_train
        self.x_test = self.x_test
        # convert from int to float
        self.x_train = self.x_train.astype('float32')
        self.x_test = self.x_test.astype('float32')
        # rescale values
        self.x_train /= 255.0
        self.x_test /= 255.0
        # Save dataset sizes
        self.train_size = self.x_train.shape[0]
        self.test_size = self.x_test.shape[0]
        # Create one hot array
        self.y_train = to_categorical(self.y_train, 10)
        self.y_test = to_categorical(self.y_test, 10)

    def data_augmentation(self): 
        self.image_generator = iaa.Sequential([iaa.Fliplr(0.5), # horizontally flip 50% of the images
            iaa.Sometimes(0.5, iaa.Affine(translate_percent={"x": (-0.15, 0.15), "y": (-0.15, 0.15)},))
        # iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*1.0), per_channel=0.5) # Add to 50%
        ])

    def next_train_batch(self, batch_size, augment=True):
        randidx = np.random.randint(self.train_size, size=batch_size)
        epoch_x = self.x_train[randidx].copy()
        epoch_y = self.y_train[randidx].copy()
        if augment:
            epoch_x = self.image_generator.augment_images(epoch_x)
        return epoch_x, epoch_y
    
    def next_test_batch(self, batch_size):
        randidx = np.random.randint(self.test_size, size=batch_size)
        epoch_x = self.x_test[randidx]
        epoch_y = self.y_test[randidx]
        return epoch_x, epoch_y

    def get_test_images(self):
        return self.x_test

    def get_test_labels(self):
        return self.y_test

    def shuffle_train(self):
        indices = np.random.permutation(self.train_size)
        self.x_train = self.x_train[indices]
        self.y_train = self.y_train[indices]

def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
    assert inputs.shape[0] == targets.shape[0]
    if shuffle:
        indices = np.arange(inputs.shape[0])
        np.random.shuffle(indices)
    for start_idx in range(0, inputs.shape[0] - batchsize + 1, batchsize):
        # if(start_idx + batchsize >= inputs.shape[0]):
        #   break;

        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], targets[excerpt]