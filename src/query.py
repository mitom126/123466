from __future__ import print_function

import sys
from color import Color
# from src.daisy import Daisy
from DB import Database
# # from src.edge import Edge
from evaluate import infer
# # from src.gabor import Gabor
# # from src.HOG import HOG
# # from src.resnet import ResNetFeat
from vggnet import VGGNetFeat

depth = 6
d_type = 'd1'
# query_idx: index cua anh truy van
query_idx = 0

if __name__ == '__main__':
    db = Database()

    # Phuong phap su dung
    methods = {
        "color": Color,
        #"daisy": Daisy,
        # "edge": Edge,
        # "hog": HOG,
        # "gabor": Gabor,
        "vgg": VGGNetFeat,
        # "resnet": ResNetFeat
    }

    try:
        mthd = sys.argv[1].lower()
    except IndexError:
        print("usage: {} <method>".format(sys.argv[0]))
        print("supported methods:\ncolor, daisy, edge, gabor, hog, vgg, resnet")

        sys.exit(1)

    # call make_samples(db) accordingly
    samples = getattr(methods[mthd](), "make_samples")(db)

    # query the first img in data.csv
    query = samples[query_idx]
    print("\n[+] query: {}\n".format(query["img"]))

    _, result = infer(query, samples=samples, depth=depth, d_type=d_type)

    for match in result:
        print("{}:\t{},\tClass {}".format(match["img"],
                                          match["dis"],
                                          match["cls"]))
