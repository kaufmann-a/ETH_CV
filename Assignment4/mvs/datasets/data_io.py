import numpy as np
import re
import sys
from PIL import Image

def read_cam_file(filename):
    intrinsics = np.zeros((3, 3))
    extrinsics = np.zeros((4, 4))
    depth_min = 0
    depth_max = 0

    with open(filename, 'r') as f:
        f.readline() #Discard as just title extrinsic
        for line in range(4):
            extrinsics[line] = [float(i) for i in f.readline().split(" ") if i.strip()]
        f.readline() # Empty line
        f.readline() # Intrinsic title
        for line in range(3):
            intrinsics[line] = [float(i) for i in f.readline().split(" ") if i.strip()]
        f.readline()
        depth_params = [float(i) for i in f.readline().split(" ")]
        depth_min = depth_params[0]
        depth_max = depth_params[1]
    return intrinsics.astype('float32'), extrinsics.astype('float32'), depth_min, depth_max

def read_img(filename):
    img = Image.open(filename)
    np_image = np.asarray(img).astype('float32')
    np_image /= 255.0
    return np_image

def read_depth(filename):
    # read pfm depth file
    return np.array(read_pfm(filename)[0], dtype=np.float32)

def read_pfm(filename):
    file = open(filename, 'rb')
    color = None
    width = None
    height = None
    scale = None
    endian = None

    header = file.readline().decode('utf-8').rstrip()
    if header == 'PF':
        color = True
    elif header == 'Pf':
        color = False
    else:
        raise Exception('Not a PFM file.')

    dim_match = re.match(r'^(\d+)\s(\d+)\s$', file.readline().decode('utf-8'))
    if dim_match:
        width, height = map(int, dim_match.groups())
    else:
        raise Exception('Malformed PFM header.')

    scale = float(file.readline().rstrip())
    if scale < 0:  # little-endian
        endian = '<'
        scale = -scale
    else:
        endian = '>'  # big-endian

    data = np.fromfile(file, endian + 'f')
    shape = (height, width, 3) if color else (height, width)

    data = np.reshape(data, shape)
    data = np.flipud(data)
    file.close()
    return data, scale


def save_pfm(filename, image, scale=1):
    file = open(filename, "wb")
    color = None

    image = np.flipud(image)

    if image.dtype.name != 'float32':
        raise Exception('Image dtype must be float32.')

    if len(image.shape) == 3 and image.shape[2] == 3:  # color image
        color = True
    elif len(image.shape) == 2 or len(image.shape) == 3 and image.shape[2] == 1:  # greyscale
        color = False
    else:
        raise Exception('Image must have H x W x 3, H x W x 1 or H x W dimensions.')

    file.write('PF\n'.encode('utf-8') if color else 'Pf\n'.encode('utf-8'))
    file.write('{} {}\n'.format(image.shape[1], image.shape[0]).encode('utf-8'))

    endian = image.dtype.byteorder

    if endian == '<' or endian == '=' and sys.byteorder == 'little':
        scale = -scale

    file.write(('%f\n' % scale).encode('utf-8'))

    image.tofile(file)
    file.close()
