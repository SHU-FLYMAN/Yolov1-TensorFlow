# 类别
CLASS = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
         'bus', 'car', 'cat', 'chair', 'cow', 'diningtable',
         'dog', 'horse', 'motorbike', 'person', 'pottedplant',
         'sheep', 'sofa', 'train', 'tvmonitor']
# 类别 >>> index
Class_to_index = {_class: _index for _index, _class in enumerate(CLASS)}
colors = [[156,102,31], [255,127,80], [255,99,71], [255,255,0], [255,153,18],
          [227,207,87], [255,255,255], [202,235,216], [192,192,192], [251,255,242],
          [160,32,240], [218,112,214], [0,255,0], [255,0,0], [25,25,112],
          [3,168,158], [128,138,135], [128,118,105], [160,82,45], [8,46,84]]
# 类别 >>> RGB
Colors_to_map = {_class: _color for _class, _color in zip(CLASS, colors)}
