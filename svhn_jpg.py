import scipy.io
import numpy
import os
from PIL import Image

def extract_dataset(filename):
    dataset = scipy.io.loadmat(filename)
    x = numpy.array(dataset['X'])
    y = dataset['y']
    x = numpy.moveaxis(x, -1, 0)
    
    for index in range(len(y)):
        temp_image = (x[index] - numpy.min(x[index])).astype(float) * 255 / (numpy.max(x[index]) - numpy.min(x[index])).astype(float)
        temp_image = numpy.round(temp_image)
        temp_image = temp_image.astype(numpy.uint8)
        image = Image.fromarray(x[index])
        image.save(os.path.join('Dataset', filename[:-4], str((y[index][0]) % 10), str(index) + '.jpg'))

    print(f'Done extracting {filename[:-4]}!')

extract_dataset('training.mat')
extract_dataset('testing.mat')
# extract_dataset('extra.mat')