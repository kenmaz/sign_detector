import turicreate as tc

imgs = tc.load_images('imgs')
imgs['label'] = imgs['path'].element_slice(5, -4)
model = tc.one_shot_object_detector.create(imgs, 'label')
model.save('sign.model')

