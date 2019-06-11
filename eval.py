import turicreate as tc
import os

m=tc.load_model('model.model')
data=tc.image_analysis.load_images('test')

p=m.predict(data['image'], confidence_threshold=0.01)

res=tc.one_shot_object_detector.util.draw_bounding_boxes(data['image'],p)
res.explore()

if not os.path.isdir('out'):
    os.makedirs('out')
for i,img in enumerate(res):
    img.save("out/out_%s.png" % i)

