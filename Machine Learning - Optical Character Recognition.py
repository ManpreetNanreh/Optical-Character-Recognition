# -*- coding: utf-8 -*-
import numpy
import pickle
import matplotlib.pyplot as plt
import scipy.linalg as la
    
def showImages(N, data, title):
    """Display the first N rows of the data as images with the title provided.
    """
    fig = plt.figure()
    fig.suptitle(title)
    
    #Assuming the grid is square. Getting the number of rows and columns 
    #for the grid in which to display the characters.
    gridSize = int(numpy.ceil(numpy.sqrt(N)))
    
    #Getting the number of rows and columns for each image in the data.
    imageSize = int(numpy.sqrt(numpy.size(data[0])))
    
    for i in range(0, N):
        im = data[i]
        reshapedIm = numpy.reshape(im, (imageSize, imageSize))
        plt.subplot(gridSize, gridSize, i+1)
        plt.axis('off')
        plt.imshow(reshapedIm, cmap='Greys', interpolation='nearest')

def fitNormal(X):
    """Return the mean (mu) and covariance (sigma) matrix determined by fitting 
       a multivariate normal distribution to the data set, X.
    """
    (M, N) = X.shape
    mu = X.sum(axis=0)/float(M)
    #Xc is the centered data matrix.
    Xc = X - mu
    sigma = numpy.dot(Xc.T, Xc)/float(M)
    return(mu, sigma)

#The goal here is to find variance for the entire data set, sigma^2 and 
#the covariance matrix for each character. We want to solve the following
#equation: beta*Sigma_d + (1-beta)*I*sigma^2, where d is each character.
#This will give the combined model for the data set.

def train(data):
    """Return the mean (mu), covariance(Sigma) and variance matrix (var) for
       the MNIST training data provided.
    """
    #Number of characters in the data set.
    d = len(data)
    #This list will hold the mean for each character.
    mu = []
    #This list will hold covariance matrices for each character.
    Sigma = []
    #This variable will hold the variance for each digit.
    vard = numpy.zeros(d)
    #This variable store the number of images for each character.
    m = numpy.zeros(d)
    
    for i in range(0,d):
        #Find the mu and Sigma for each character.
        [mui, Si] = fitNormal(data[i])
        mu.append(mui)
        Sigma.append(Si)
        #Find the variance for each character.
        vard[i] = numpy.mean(numpy.diag(Si))
        #Number of images for each character.
        m[i] = numpy.shape(data[i])[0]
        
    #Solving the equation: sigma^2 = Sum_d (m_d * sigma_d^2 / m).
    var = numpy.sum(vard*m)/numpy.sum(m)
    return(mu, Sigma, var)

#The data provided is a set of matrices.
def flatten(data):
    """Return the test data (X) by combining the data into a single matrix 
       containing all the test data and vector (Y) labeling the test data.
    """
    #Number of characters in the data set.
    d = len(data)
    #X is the matrix containing all test data.
    X = numpy.vstack(data)
    M = numpy.shape(X)[0]
    #Y is a vector indicating the class of each test character.
    Y = numpy.zeros(M, dtype='int')
    m1 = 0
    m2 = 0
    
    for i in range(0,d):
        #This is the number of images in the matrix.
        m = numpy.shape(data[i])[0]
        m2 = m2 + m
        Y[m1:m2] = i
        m1 = m1 + m

    return(X,Y)

def combine(Sigma, var, beta):
    """Return the combined model by combining the complex model and the simple
       model. Sigma is the set of complex models, var is the variance of the
       simple model and bets is the weight of the complex models.
    """
    #Number of complex models from Sigma.
    d = len(Sigma)
    N = numpy.shape(Sigma[0])[0]
    CombinedModel = []
    SimpleModel = (1-beta)*var*numpy.eye(N)
    
    for i in range(0, d):
        #This is the combined model for character i.
        ComplexModel = beta*Sigma[i] + SimpleModel
        CombinedModel.append(ComplexModel)
    
    return(CombinedModel)

def logMVNchol(X, mu, Sigma):
    """Return the log probabilities for the multivariate normal distribution.
       Uses the Cholesky factorization for the covariance matrix.
       Formula being solved here (in Latex): 
       -\frac{1}{2}(x-mu)^T*Sigma^{-1}*(x-mu)-\frac{n}{2}log(2*pi)-\frac{1}{2}log(|Sigma|)
    """
    (M,N) = X.shape
    #logProb is the log-probability matrix.
    logProb = numpy.zeros(M)
    #Find the Cholesky factorization for the covariance matrix.
    L = la.cholesky(Sigma, lower=True)
    
    #Precompute parts of the formula.
    logDet = numpy.linalg.slogdet(Sigma)[1]
    logAlpha = numpy.log(2*numpy.pi)*N/2. + logDet
    
    for i in range(0, M):
        #This is (X - mu)
        x = X[i,:] - mu
        
        #Now compute x*sigmainv*x^T using Cholesky factorization.
        y = la.solve_triangular(L, x, lower=True, check_finite=False)
        logProb[i] = -numpy.dot(y,y)/2. - logAlpha
        
    return(logProb)

def predict(X, mu, Sigma):
    """Return the log probabilities for each character hence giving the 
       soft prediction for each image in data matrix X.
    """
    #M is the number of images.
    M = numpy.shape(X)[0]
    #The number of characters given.
    d = len(mu)
    #LogProb is the log-probability matrix.
    logProb = numpy.zeros((M, d))

    for i in range(0, d):
        #Compute the log-probabilities for all images for character i.
        logProb[:, i] = logMVNchol(X, mu[i], Sigma[i])
        
    return(logProb)

def evaluate(logProb, X, Y):
    """Evaluate the hard prediction from soft predictions, determine the accuracy,
       and display the correct and incorrect predictions.
    """
    #Here, convert the soft predictions into hard predictions.
    Yhat = numpy.argmax(logProb, axis=1)
    #Compare Yhat to Y and see which predictions are correct.
    #Store their index.
    correctIndex = (Y == Yhat)
    
    #Calculate the percentage of correct prediction.
    M = numpy.shape(X)[0]
    accuracy = 100*numpy.sum(correctIndex)/float(M)
    print "Prediction accuracy: ", accuracy
    
    #Want to display 50 Images.
    N = 50
    #These are the correctly clasified images.
    correct = X[correctIndex,:]

    #Show random Images.
    numpy.random.shuffle(correct)
    showImages(N, correct, 'Correctly clasified Images.')
    
    #Show N random misclassified Images.
    #These the are the misclassified images.
    errors = X[~correctIndex,:]
    numpy.random.shuffle(errors)
    showImages(N, errors, 'Misclassified Images.')    
    
def ocr():
    """Read the MNIST data, train the training data, call prediction on testing
       data and evaluate the predictions.
    """
    #Read the mnist data from the file.
    fileName = 'mnist.pickle' #Put the filename here
    with open(fileName, 'rb') as f:
        data = pickle.load(f)
        
    #Train the training data
    (mu, Sigma, var) = train(data['training'])
    print('Training Finished')
    
    #Predict the probabilities.
    (X,Y) = flatten(data['testing'])
    #Using beta of 0.25 for best prediction.
    beta = 0.25
    SigmaN = combine(Sigma, var, beta)
    logProb = predict(X, mu, SigmaN)
    print('Prediction Finished')
    
    evaluate(logProb, X, Y)
    print('Evaluation Finished')