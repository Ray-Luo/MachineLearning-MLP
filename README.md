# MachineLearning-MLP
### 1. What is the math behind like?
![Math for MLP](https://github.com/Ray-Luo/MachineLearning-MLP/blob/master/Math.jpg?raw=true)

### 2. Class diagram for MLP
![Class diagram](https://github.com/Ray-Luo/MachineLearning-MLP/blob/master/ClassDiagram.jpg?raw=true)

### 3. How to use it?
```
# Declare a MLP object
mlp = MLP()

# input and target data
x = np.linspace(0,1,40).reshape((40,1))
x = (x-0.5)*2
y = np.sin(2*np.pi*x) + np.cos(4*np.pi*x) + np.random.randn(40).reshape((40,1))*0.2
train = x[0::2,:]
test = x[1::4,:]
valid = x[3::4,:]
traintarget = y[0::2,:]
testtarget = y[1::4,:]
validtarget = y[3::4,:]

# declare layers
X = Layer(value = train)
target = Layer(value = traintarget)
h1 = Layer(rows = X.rows, cols = 3)        # first hidden layer
h2 = Layer(rows = X.rows, cols = 2)        # 2nd hidden layer
h3 = Layer(rows = X.rows, cols = 1)        # 3rd hidden layer
output = Layer(rows = X.rows, cols = target.cols)

# connect layers
X.connect(h1)
h1.connect(h2)
h2.connect(h3)
h3.connect(output)

# add layers to MLP and train
mlp.addLayers([X, h1,
               h2,
               h3,
               output, target])
mlp.train()

# predict
test_x = np.ones((1,1))
output = mlp.predict(x=test_x)
print(output)
```
