import turicreate as tc

m=tc.load_model('model.model')
m.export_coreml('Model.mlmodel')

