import idx2numpy
import gzip


pathToTrainImages = "./samples/train-images-idx3-ubyte.gz"
pathToTrainLabels = "./samples/train-labels-idx1-ubyte.gz"
pathToTestImages = "./samples/t10k-images-idx3-ubyte.gz"
pathToTestLabels = "./samples/t10k-labels-idx1-ubyte.gz"


# Importing the datasets
train_images_import = gzip.open(pathToTrainImages, "r")
train_labels_import = gzip.open(pathToTrainLabels, "r")

test_images_import = gzip.open(pathToTestImages, "r")
test_labels_import = gzip.open(pathToTestLabels, "r")


# Converting them with idx2numpy
train_images = idx2numpy.convert_from_file(train_images_import)
train_labels = idx2numpy.convert_from_file(train_labels_import)

test_images = idx2numpy.convert_from_file(test_images_import)
test_labels = idx2numpy.convert_from_file(test_labels_import)
