import turicreate as tc
m=tc.load_model('sign.model')
data=tc.image_analysis.load_images('test')
p=m.predict(data['image'], confidence_threshold=0.001, iou_threshold=1)
res=tc.one_shot_object_detector.util.draw_bounding_boxes(data['image'],p)
res.explore()
