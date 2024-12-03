from base import test
from MODEL import model

model.load_weights('models/checkpoint')
test_data = test.as_numpy_iterator()
sample = test_data.next()
yhat = model.predict(sample[0])
print('~'*100, 'REAL TEXT')




