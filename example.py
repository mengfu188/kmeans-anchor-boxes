import glob
import xml.etree.ElementTree as ET

import numpy as np

from kmeans import kmeans, avg_iou

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--ann-path', default='Annotations')
parser.add_argument('--clusters', nargs='+', type=int, default=[1, 2, 3, 4, 5, 6, 7, 8, 9])
parser.add_argument('--img-size', type=int, default=range(320, 609, 32), nargs='+')
args = parser.parse_args()

ANNOTATIONS_PATH = args.ann_path
CLUSTERS = args.clusters


def load_dataset(path):
    dataset = []
    for xml_file in glob.glob("{}/*xml".format(path)):
        tree = ET.parse(xml_file)

        height = int(tree.findtext("./size/height"))
        width = int(tree.findtext("./size/width"))

        for obj in tree.iter("object"):
            xmin = int(float(obj.findtext("bndbox/xmin"))) / width
            ymin = int(float(obj.findtext("bndbox/ymin"))) / height
            xmax = int(float(obj.findtext("bndbox/xmax"))) / width
            ymax = int(float(obj.findtext("bndbox/ymax"))) / height

            dataset.append([xmax - xmin, ymax - ymin])

    return np.array(dataset)


data = load_dataset(ANNOTATIONS_PATH)
for cluster in CLUSTERS:
    print('-' * 20)
    print(f'cluster num is {cluster}')
    out = kmeans(data, k=cluster)
    print("Accuracy: {:.2f}%".format(avg_iou(data, out) * 100))
    print("Boxes:\n {}".format(out))
    for size in args.img_size:
        print(f'for image size {size}, Boxes:')
        print(out * size)

    ratios = np.around(out[:, 0] / out[:, 1], decimals=2).tolist()
    print("Ratios:\n {}".format(sorted(ratios)))
