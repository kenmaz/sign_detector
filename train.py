import turicreate as tc
tc.config.set_runtime_config('TURI_DEFAULT_NUM_PYLAMBDA_WORKERS', 20)
imgs = tc.load_images('imgs')
imgs['label'] = imgs['path'].element_slice(5, -4)
model = tc.one_shot_object_detector.create(imgs, 'label')
model.save('model.model')

