import json
import os

name_map = {
        "train": "train",
        "dev": "valid",
        "test1": "test"
        }


os.makedirs("nlvr2", exist_ok=True)
for name0, name1 in name_map.items():
    dataset = []
    for i, line in enumerate(open("%s.json" % name0)):
        raw_data = json.loads(line)
        image_id = "-".join(raw_data["identifier"].split('-')[:-1])
        image_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'images', name0)
        if name0 == 'train':
            image_path = os.path.join(image_path, str(raw_data['directory']))
        img0_path = os.path.join(image_path, image_id + "-img0.png")
        img1_path = os.path.join(image_path, image_id + "-img1.png")
        sents = [raw_data["sentence"]]
        new_data = {
                'uid': "%s_%s_%d" % ('nlvr2', name1, i),
                'img0': img0_path,
                'img1': img1_path,
                'sents': sents,
                'identifier': raw_data['identifier'],
                'label': raw_data['label'],
                }
        dataset.append(new_data)
    json.dump(dataset, open("nlvr2/%s.json" % name1, 'w'),
              indent=4, sort_keys=True)
