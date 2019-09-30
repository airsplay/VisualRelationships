import json
import os

name_map = {
        "train": "train",
        "val": "valid",
        "test": "test"
        }

def find_full_name(path):
    if os.path.exists(path + ".jpg"):
        return path + ".jpg"
    elif os.path.exists(path + ".png"):
        return path + ".png"
    else:
        assert False, path

image_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'resized_images')
os.makedirs("spotdiff", exist_ok=True)
for name0, name1 in name_map.items():
    data = json.load(open("spotdiff/data/annotations/%s.json" % name0))
    dataset = []
    for i, datum in enumerate(data):
        image_id = datum['img_id']
        sents = datum['sentences']
        img0_name = "%s" % image_id
        img1_name = "%s_2" % image_id
        imgdiff_name = "%s_diff" % image_id
        new_data =  {
            'uid': "%s_%s_%d" % ('spotdiff', name1, i),
            'img0': find_full_name(os.path.join(image_path, img0_name)),
            'img1': find_full_name(os.path.join(image_path, img1_name)),
            'sents': sents,
            'imgdiff': find_full_name(os.path.join(image_path, imgdiff_name)),
        }
        dataset.append(new_data)
    json.dump(dataset, open("spotdiff/%s.json" % name1, 'w'),
              indent=4, sort_keys=True)
