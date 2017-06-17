import os
import tarfile
from six.moves import urllib


def unpack(filename, filepath, extract_to=None):
    f_ext = (filename.split('.')[-1]).lower()
    file_with_path = os.path.join(filepath, filename)
    extract_to = filepath if extract_to is None else extract_to
    if f_ext == 'tgz':
        print "Extract file:", filename
        with tarfile.open(file_with_path) as tf:
            tf.extractall(path=extract_to)


def download(url, filepath, filename):
    full_url = url+'/'+filepath+'/'+filename
    print "Full URL:", full_url
    print "Filepath:", filepath

    if not os.path.isdir(filepath):
        print "Directory {} does not exist, create ..". format(filepath)
        os.makedirs(filepath)

    file_with_path = os.path.join(filepath, filename)
    urllib.request.urlretrieve(full_url, file_with_path)
    unpack(filename, filepath)

