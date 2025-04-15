import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder

# -------------------- NN for Pipeline --------------------

class myNN:
    """
    Standard Feed-Forward Neural Network classifier for comparison.
    
    This implements a three-class classifier using a neural network.
    """
    
    def __init__(self, input_size=4, output_size=3):
        """
        Initialize the neural network classifier.
        
        Parameters:
        -----------
        input_size : int
            Number of input features
        output_size : int
            Number of output classes
        """
        # TODO: Initialize network architecture and parameters
        self.input_size = input_size
        self.output_size = output_size
        
        # Add other necessary attributes for your implementation
        self.w1, self.b1, self.w2, self.b2 = self.init_params((10, 4), (3, 10), (10, 1), (3, 1))
        self.epochs = 200
        self.lr = 0.01

        return

    # ------------------------------ Extra function additions ------------------------------

    def init_params(self, w1shape, w2shape, b1shape, b2shape):
        w1 = np.random.random(w1shape)
        w2 = np.random.random(w2shape)
        b1 = np.random.random(b1shape)
        b2 = np.random.random(b2shape)
        return w1, b1, w2, b2
    
    def OHE(self, y:np.ndarray):
        res = np.zeros((y.shape[0], self.output_size))
        for i in range(y.shape[0]):
            res[i][y[i]] = 1
        return res  # dim: (y.shape[0], 3)

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

	# ---------------------------------------------------------------------------------------

    def forward(self, X):
        """
        Forward propagation through the network.
        
        Parameters:
        -----------
        X : numpy.ndarray
            Input feature data
            
        Returns:
        --------
        numpy.ndarray
            Output probabilities for each class
        """
        # TODO: Implement forward propagation
        
        self.z1 = self.w1 @ X + self.b1
        self.a1 = self.sigmoid(self.z1)
        self.z2 = self.w2 @ self.a1 + self.b2
        self.a2 = self.sigmoid(self.z2)
        
        return self.a2
        
    def backward(self, x, y, output):
        """
        Backward propagation to update weights.
        
        Parameters:
        -----------
        X : numpy.ndarray
            Input feature data
        y : numpy.ndarray
            True labels 
        output : numpy.ndarray
            Output from forward propagation
        """
        # TODO: Implement backpropagation
        
        # LOSS: TSS
        dz2 = (self.a2 - y) * self.a2 * (1 - self.a2)
        dw2 = dz2 @ self.a1.T
        db2 = np.sum(dz2, axis=1, keepdims=True)
        dz1 = self.w2.T @ dz2 * self.a1 * (1 - self.a1)
        dw1 = dz1 @ x.T
        db1 = np.sum(dz1, axis=1, keepdims=True)

        # changing weights
        self.w1 -= self.lr * dw1
        self.b1 -= self.lr * db1
        self.w2 -= self.lr * dw2
        self.b2 -= self.lr * db2
        
        return dz2

    def fit(self, X, y):
        """
        Train the neural network using backpropagation.
        
        Parameters:
        -----------
        X : numpy.ndarray
            Training feature data
        y : numpy.ndarray
            Training labels
        """
        # TODO: Implement network training

        errors = []
        xtrain = X.T
        ytrain = self.OHE(y).T

        # train the model
        for i in range(self.epochs):
            self.forward(xtrain)                                    # forward pass
            error = np.sum(self.backward(xtrain, ytrain, None))     # backward pass, my implementation don't need the third argument
            errors.append(error)

        return errors

        
    def predict(self, X):
        """
        Predict class labels for samples in X.
        
        Parameters:
        -----------
        X : numpy.ndarray
            Input feature data
            
        Returns:
        --------
        numpy.ndarray
            Predicted class labels
        """
        # TODO: Implement prediction

        xtest = X.T
        # once we are trained, prediction is just a forward pass
        pred = self.forward(xtest)
        return pred

# ------------------------------------------------------------



class ClassificationPipeline:
    """
    Two-stage classification pipeline: SVM followed by FFNN.
    
    The SVM classifier determines the likelihood of one class from the rest.
    The FFNN determines probabilities for the remaining two classes.
    """
    
    def __init__(self, svm_target=0):
        """
        Initialize the classification pipeline.
        
        Parameters:
        -----------
        svm_target : int
            Specifies the target class for the SVM (0, 1, or 2).
        """
        # Store the target class for the SVM
        self.svm_target = svm_target
        
        # TODO: Initialize the SVM classifier
        self.svm = SVC(kernel = "rbf", probability = True)
        
        # TODO: Initialize attributes for FFNN
        # You may add additional attributes as needed for your implementation
        self.FFNN = myNN()
    
    # ----------------------------------------------------------------------------------------------------
    def makeBinaryLabels(self, y: np.ndarray):
        """
        Takes raw target array and convert into binary targets (1 for correct class, 0 for wrong class)
        """
        res = [1 if ele == self.svm_target else 0 for ele in y]
        res = np.array(res)
        return res
    # ----------------------------------------------------------------------------------------------------

        
    def fitSVM(self, X, y):
        """
        Convert the original labels into binary labels and fit the SVM.
        
        Parameters:
        -----------
        X : numpy.ndarray
            Training feature data
        y : numpy.ndarray
            Training labels
        """
        # TODO: Convert labels to binary (1 for target class, 0 for others)
        # and fit the SVM model

        xtrain = X
        ytrain = self.makeBinaryLabels(y)
        self.svm.fit(xtrain, ytrain)

        return
        
    def probsSVM(self, X):
        """
        Return the probability estimates from SVM.
        
        Parameters:
        -----------
        X : numpy.ndarray
            Input feature data
            
        Returns:
        --------
        numpy.ndarray
            Probability that each input belongs to the target class
        """
        # TODO: Return probabilities from SVM
        return self.svm.predict_proba(X)[:, 1]     # probability of being from positive class
        
    def fitFFNN(self, X, y):
        """
        Train the FFNN component of the pipeline.
        
        Parameters:
        -----------
        X : numpy.ndarray
            Training feature data
        y : numpy.ndarray
            Training labels
        """

        # TODO: Implement FFNN training with backpropagation
        self.ffnnErrors = self.FFNN.fit(X, y)
        return
        
        
    def probsFFNN(self, X):
        """
        Return the probability estimates from FFNN.
        
        Parameters:
        -----------
        X : numpy.ndarray
            Input feature data
            
        Returns:
        --------
        numpy.ndarray
            Probability estimates for the two classes other than the target class
        """

        # TODO: Return probabilities from FFNN for the other two classes
        return self.FFNN.predict(X)


    def fit(self, X, y):
        """
        Fit both SVM and FFNN components of the pipeline.
        
        Parameters:
        -----------
        X : numpy.ndarray
            Training feature data
        y : numpy.ndarray
            Training labels
        """
        # TODO: Implement pipeline fitting
        # using the fitSVM and fitFFNN functions

        self.fitSVM(X, y)       # train svm
        self.fitFFNN(X, y)      # train NN

        return
        
    def predict(self, X):
        """
        Predict class labels for samples in X.
        
        Parameters:
        -----------
        X : numpy.ndarray(one-hot encoded)
            Input feature data
            
        Returns:
        --------
        numpy.ndarray
            Predicted class labels
        """
        # TODO: Implement prediction using both SVM and FFNN
        # Combine the probabilities and determine the final class prediction

        predSVM  = self.probsSVM(X)         # a 1d array having prob of being from the correct class
        predFFNN = self.probsFFNN(X).T      # a (n x 3) array having probability of being from each class

        res = np.zeros((X.shape[0], ))
        
        for i in range(X.shape[0]):
            # if svm is sure of a class, listen to it
            if predSVM[i] > 0.5:
                res[i] = self.svm_target
            # otherwise seek answer from the predictions of the NN
            else:
                res[i] = np.argmax(predFFNN[i])
                    
        return res
        
        
        
class StandardNNClassifier:
    """
    Standard Feed-Forward Neural Network classifier for comparison.
    This implements a three-class classifier using a neural network.
    """
    
    def __init__(self, input_size=4, output_size=3):
        """
        Initialize the neural network classifier.
        
        Parameters:
        -----------
        input_size : int
            Number of input features
        output_size : int
            Number of output classes
        """
        # TODO: Initialize network architecture and parameters
        self.input_size = input_size
        self.output_size = output_size
        
        # Add other necessary attributes for your implementation
        self.w1, self.b1, self.w2, self.b2 = self.init_params((10, 4), (3, 10), (10, 1), (3, 1))
        self.epochs = 200
        self.lr = 0.01

        return

    # ------------------------------ Extra function additions ------------------------------

    def init_params(self, w1shape, w2shape, b1shape, b2shape):
        w1 = np.random.random(w1shape)
        w2 = np.random.random(w2shape)
        b1 = np.random.random(b1shape)
        b2 = np.random.random(b2shape)
        return w1, b1, w2, b2
    
    def OHE(self, y:np.ndarray):
        res = np.zeros((y.shape[0], self.output_size))
        for i in range(y.shape[0]):
            res[i][y[i]] = 1
        return res  # dim: (y.shape[0], 3)

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    
    # --------------------------------------------------------------------------------------

    def forward(self, X):
        """
        Forward propagation through the network.
        
        Parameters:
        -----------
        X : numpy.ndarray
            Input feature data
            
        Returns:
        --------
        numpy.ndarray
            Output probabilities for each class
        """
        # TODO: Implement forward propagation
        
        self.z1 = self.w1 @ X + self.b1
        self.a1 = self.sigmoid(self.z1)
        self.z2 = self.w2 @ self.a1 + self.b2
        self.a2 = self.sigmoid(self.z2)
        
        return self.a2
        
    def backward(self, x, y, output):
        """
        Backward propagation to update weights.
        
        Parameters:
        -----------
        X : numpy.ndarray
            Input feature data
        y : numpy.ndarray
            True labels 
        output : numpy.ndarray
            Output from forward propagation
        """
        # TODO: Implement backpropagation
        
        # LOSS: TSS
        dz2 = (self.a2 - y) * self.a2 * (1 - self.a2)
        dw2 = dz2 @ self.a1.T
        db2 = np.sum(dz2, axis=1, keepdims=True)
        dz1 = self.w2.T @ dz2 * self.a1 * (1 - self.a1)
        dw1 = dz1 @ x.T
        db1 = np.sum(dz1, axis=1, keepdims=True)

        # changing weights
        self.w1 -= self.lr * dw1
        self.b1 -= self.lr * db1
        self.w2 -= self.lr * dw2
        self.b2 -= self.lr * db2
        
        return dz2

    def fit(self, X, y):
        """
        Train the neural network using backpropagation.
        
        Parameters:
        -----------
        X : numpy.ndarray
            Training feature data
        y : numpy.ndarray
            Training labels
        """
        # TODO: Implement network training

        errors = []
        xtrain = X.T
        ytrain = self.OHE(y).T

        # train the model
        for i in range(self.epochs):
            self.forward(xtrain)                                    # forward pass
            error = np.sum(self.backward(xtrain, ytrain, None))     # backward pass, my implementation don't need the third argument
            errors.append(error)

        return errors

        
    def predict(self, X):
        """
        Predict class labels for samples in X.
        
        Parameters:
        -----------
        X : numpy.ndarray
            Input feature data
            
        Returns:
        --------
        numpy.ndarray
            Predicted class labels
        """
        # TODO: Implement prediction

        xtest = X.T
        # once we are trained, prediction is just a forward pass
        pred = self.forward(xtest)
        res = np.zeros((xtest.shape[1], ))
        for i in range(xtest.shape[1]):
            res[i] = np.argmax(pred.T[i])
        
        return res