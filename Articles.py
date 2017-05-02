import cPickle
import numpy as np 
from util_functions import featureUniform, gaussianFeature, fileOverWriteWarning
from random import sample, randint, uniform
import json
import time
from random import *

class Article():	
	def __init__(self, aid, atype, FV=None, rFV = None):
		self.id = aid
		self.featureVector = FV
		#self.articles
		self.returnFeatureVector = rFV
		self.type = atype
		

class ArticleManager():
	def __init__(self, dimension, n_articles, ArticleGroups, FeatureFunc, argv, userFeature_theta, userFeature_beta ):
		self.signature = "Article manager for simulation study"
		self.dimension = dimension
		self.n_articles = n_articles
		self.ArticleGroups = ArticleGroups
		self.FeatureFunc = FeatureFunc
		self.thetaFunc = FeatureFunc
		self.betaFunc = FeatureFunc
		self.argv = argv

		self.signature = "A-"+str(self.n_articles)+"+AG"+ str(self.ArticleGroups)+"+TF-"+self.FeatureFunc.__name__

		self.userFeature_theta = userFeature_theta
		self.userFeature_beta = userFeature_beta
	def saveArticles(self, Articles, filename, force = False):
		with open(filename, 'w') as f:
			for i in range(len(Articles)):
				f.write(json.dumps((Articles[i].id, Articles[i].type, Articles[i].featureVector, Articles[i].returnFeatureVector)) + '\n')


	def loadArticles(self, filename):
		articles = []
		with open(filename, 'r') as f:
			for line in f:
				aid, atype, featureVector, returnFeatureVector = json.loads(line)
				articles.append(Article(aid, atype,  np.array(featureVector), np.array(returnFeatureVector)))
		return articles

	#automatically generate masks for articles, but it may generate same masks
	def generateMasks(self):
		mask = {}
		for i in range(self.ArticleGroups):
			mask[i] = np.random.randint(2, size = self.dimension)
		return mask

	def simulateArticlePool(self):
		articles = []
		
		articles_id = {}
		mask = self.generateMasks()

		for i in range(self.ArticleGroups):
			articles_id[i] = range((self.n_articles*i)/self.ArticleGroups, (self.n_articles*(i+1))/self.ArticleGroups)

			for key in articles_id[i]:
				featureVector = np.multiply(featureUniform(self.dimension, {}), mask[i])
				l2_norm = np.linalg.norm(featureVector, ord =2)
				articles.append(Article(key, featureVector/l2_norm ))

	
		return articles



	def largeProduct(self, featureVector):
		vector = self.FeatureFunc(self.dimension, {})
		vector_l2_norm = np.linalg.norm(vector, ord =2)

		while np.dot(vector, featureVector) < 1 or vector_l2_norm >4:
			vector = self.FeatureFunc(self.dimension, {})
			vector_l2_norm = np.linalg.norm(vector, ord =2)
			#print vector, vector_l2_norm
		print 'large', np.dot(vector, featureVector), vector_l2_norm
		return vector
	def smallProduct(self, featureVector):
		vector = self.FeatureFunc(self.dimension, {})
		vector_l2_norm = np.linalg.norm(vector, ord =2)
		while np.dot(vector, featureVector) > 0 or np.dot(vector, featureVector)<-0.1 or vector_l2_norm >4: #Make it between -0.1 and 0 to make sure the variance is not too large
			vector = self.FeatureFunc(self.dimension, {})
			vector_l2_norm = np.linalg.norm(vector, ord =2)
		print 'small', np.dot(vector, featureVector), vector_l2_norm
		return vector

	
	def largeProduct_Uniform(self, featureVector):
		vector = np.array([random() for _ in range(self.dimension)])
		vector_l2_norm = np.linalg.norm(vector, ord =2)
		final_vector_norm = np.asarray(vector)/float(vector_l2_norm)
		vector_l2_norm = np.linalg.norm(final_vector_norm, ord =2)
		#final_vector_norm = vector

		while np.dot(final_vector_norm, featureVector) < 0.5 or vector_l2_norm >1:
			vector = np.array([random() for _ in range(self.dimension)])
			vector_l2_norm = np.linalg.norm(vector, ord =2)
			#final_vector_norm = np.asarray(vector)/float(vector_l2_norm)
			final_vector_norm = vector

			#ratio = np.dot(vector, featureVector)/uniform(3, 10)
			ratio =1
			final_vector_norm = vector/ratio
			vector_l2_norm = np.linalg.norm(final_vector_norm, ord =2)
		print 'large', np.dot(final_vector_norm, featureVector), vector_l2_norm
		return final_vector_norm
	def smallProduct_Uniform(self, featureVector):
		'''
		vector = np.array([random() for _ in range(self.dimension-1)])
		
		lastDim = (0.3 - np.dot(vector, featureVector[0:self.dimension-1]))/float(featureVector[-1])
		wholevector = vector.tolist() + [lastDim]
		vector = np.asarray(wholevector) 
		vector_l2_norm = np.linalg.norm(vector, ord =2)
		final_vector_norm = np.asarray(vector)/float(vector_l2_norm)
		print  np.dot(final_vector_norm, featureVector)
		'''
		vector = np.array([random() for _ in range(self.dimension)])
		vector_l2_norm = np.linalg.norm(vector, ord =2)
		final_vector_norm = np.asarray(vector)/float(vector_l2_norm)
		while np.dot(final_vector_norm, featureVector) > -0.5 or vector_l2_norm >1:
			'''
			vector = np.array([random() for _ in range(self.dimension-1)])
			lastDim = (0.7 - np.dot(vector, featureVector[0:self.dimension-1]))/float(featureVector[-1])
			wholevector = vector.tolist() + [lastDim]
			vector = np.asarray(wholevector) 
			vector_l2_norm = np.linalg.norm(vector, ord =2)
			final_vector_norm = np.asarray(vector)/float(vector_l2_norm)
			'''
			vector = np.array([random() for _ in range(self.dimension)])
			vector_l2_norm = np.linalg.norm(vector, ord =2)
			final_vector_norm = vector
			
			#ratio = np.dot(vector, featureVector)/uniform(0, 0.2)
			ratio = 1
			final_vector_norm = vector/ratio
			vector_l2_norm = np.linalg.norm(final_vector_norm, ord =2)
			print np.dot(final_vector_norm, featureVector), vector_l2_norm
			
		print 'small', np.dot(final_vector_norm, featureVector), vector_l2_norm
		return final_vector_norm

	def largeProduct_BAK2(self, featureVector):
		vector = np.array([random() for _ in range(self.dimension)])
		vector_l2_norm = np.linalg.norm(vector, ord =2)
		final_vector_norm = np.asarray(vector)/float(vector_l2_norm)
		vector_l2_norm = np.linalg.norm(final_vector_norm, ord =2)
		#final_vector_norm = vector

		while np.dot(final_vector_norm, featureVector) < 2:
			vector = np.array([random() for _ in range(self.dimension)])
			vector_l2_norm = np.linalg.norm(vector, ord =2)
			#final_vector_norm = np.asarray(vector)/float(vector_l2_norm)
			final_vector_norm = 1*vector

			#ratio = np.dot(vector, featureVector)*uniform(3, 10)
			#ratio =1
			#final_vector_norm = vector/ratio
			#print np.dot(final_vector_norm, featureVector)
			vector_l2_norm = np.linalg.norm(final_vector_norm, ord =2)

		print 'large', np.dot(final_vector_norm, featureVector), vector_l2_norm
		return final_vector_norm
	def smallProduct_BAK2(self, featureVector):
		'''
		vector = np.array([random() for _ in range(self.dimension-1)])
		
		lastDim = (0.3 - np.dot(vector, featureVector[0:self.dimension-1]))/float(featureVector[-1])
		wholevector = vector.tolist() + [lastDim]
		vector = np.asarray(wholevector) 
		vector_l2_norm = np.linalg.norm(vector, ord =2)
		final_vector_norm = np.asarray(vector)/float(vector_l2_norm)
		print  np.dot(final_vector_norm, featureVector)
		'''
		vector = np.array([random() for _ in range(self.dimension)])
		vector_l2_norm = np.linalg.norm(vector, ord =2)
		final_vector_norm = np.asarray(vector)/float(vector_l2_norm)
		while np.dot(final_vector_norm, featureVector) > 0.4 or np.dot(final_vector_norm, featureVector) <0.2:   #or vector_l2_norm >1
			'''
			vector = np.array([random() for _ in range(self.dimension-1)])
			lastDim = (0.7 - np.dot(vector, featureVector[0:self.dimension-1]))/float(featureVector[-1])
			wholevector = vector.tolist() + [lastDim]
			vector = np.asarray(wholevector) 
			vector_l2_norm = np.linalg.norm(vector, ord =2)
			final_vector_norm = np.asarray(vector)/float(vector_l2_norm)
			'''
			vector = np.array([random() for _ in range(self.dimension)])
			vector_l2_norm = np.linalg.norm(vector, ord =2)
			final_vector_norm = 0.2*vector
			#print np.dot(final_vector_norm, featureVector)
			#ratio = np.dot(vector, featureVector)/uniform(0, 1)
			ratio = 1
			#final_vector_norm = vector/ratio
			vector_l2_norm = np.linalg.norm(final_vector_norm, ord =2)
			print np.dot(final_vector_norm, featureVector) 
			
		print 'small', np.dot(final_vector_norm, featureVector), vector_l2_norm
		return final_vector_norm


	def largeProduct_Exp(self, featureVector):
		vector = np.array([random() for _ in range(self.dimension)])
		vector_l2_norm = np.linalg.norm(vector, ord =2)
		final_vector_norm = np.asarray(vector)/float(vector_l2_norm)
		vector_l2_norm = np.linalg.norm(final_vector_norm, ord =2)
		#final_vector_norm = vector

		while np.exp(np.dot(final_vector_norm, featureVector) )< 3 and vector_l2_norm >1:
			vector = np.array([random() for _ in range(self.dimension)])
			vector_l2_norm = np.linalg.norm(vector, ord =2)
			final_vector_norm = np.asarray(vector)/float(vector_l2_norm)
			final_vector_norm = vector

			ratio = np.exp(np.dot(vector, featureVector))/uniform(3, 5)
			final_vector_norm = vector/ratio
			vector_l2_norm = np.linalg.norm(final_vector_norm, ord =2)
		print 'large', np.exp(np.dot(final_vector_norm, featureVector)), vector_l2_norm
		return final_vector_norm
	def smallProduct_Exp(self, featureVector):
		'''
		vector = np.array([random() for _ in range(self.dimension-1)])
		
		lastDim = (0.3 - np.dot(vector, featureVector[0:self.dimension-1]))/float(featureVector[-1])
		wholevector = vector.tolist() + [lastDim]
		vector = np.asarray(wholevector) 
		vector_l2_norm = np.linalg.norm(vector, ord =2)
		final_vector_norm = np.asarray(vector)/float(vector_l2_norm)
		print  np.dot(final_vector_norm, featureVector)
		'''
		vector = np.array([random() for _ in range(self.dimension)])
		vector_l2_norm = np.linalg.norm(vector, ord =2)
		final_vector_norm = np.asarray(vector)/float(vector_l2_norm)
		while np.exp(np.dot(final_vector_norm, featureVector)) > 1.2 and vector_l2_norm >1:
			'''
			vector = np.array([random() for _ in range(self.dimension-1)])
			lastDim = (0.7 - np.dot(vector, featureVector[0:self.dimension-1]))/float(featureVector[-1])
			wholevector = vector.tolist() + [lastDim]
			vector = np.asarray(wholevector) 
			vector_l2_norm = np.linalg.norm(vector, ord =2)
			final_vector_norm = np.asarray(vector)/float(vector_l2_norm)
			'''
			vector = np.array([random() for _ in range(self.dimension)])
			vector_l2_norm = np.linalg.norm(vector, ord =2)
			final_vector_norm = vector
			#print np.dot(final_vector_norm, featureVector)
			ratio = np.exp(np.dot(vector, featureVector))/uniform(0, 1.2)
			final_vector_norm = vector/ratio
			vector_l2_norm = np.linalg.norm(final_vector_norm, ord =2)
			
		print 'small', np.exp(np.dot(final_vector_norm, featureVector)), vector_l2_norm
		return final_vector_norm




	def simulateArticlePool_2SetOfFeature(self):
		articlesDic = {}		
		articles_id = {}
		centroids = [0.2, 0.9]
		
		articlesDic['small_small'] = []
		articlesDic['small_large'] = []
		articlesDic['large_small'] = []
		articlesDic['large_large'] = []

		for i in range(self.n_articles):
			#print self.userFeature_beta
			print(self.n_articles)
			print('article manager iter ' + str(i))
			small_theta = self.smallProduct(self.userFeature_theta)
			small_beta = self.smallProduct(self.userFeature_beta)
			large_theta = self.largeProduct(self.userFeature_theta)
			large_beta = self.largeProduct(self.userFeature_beta)
			'''
			small_theta = self.smallProduct_Exp(self.userFeature_theta)
			small_beta = self.smallProduct_Exp(self.userFeature_beta)
			large_theta = self.largeProduct_Exp(self.userFeature_theta)
			large_beta = self.largeProduct_Exp(self.userFeature_beta)
			'''

			articlesDic['small_small'].append(Article(4*i, 'smallTheta_smallBeta', small_theta, small_beta))
			articlesDic['small_large'].append(Article(4*i+1, 'smallTheta_largeBeta',small_theta, large_beta))
			articlesDic['large_small'].append(Article(4*i+2, 'largeTheta_smallBeta', large_theta, small_beta))
			articlesDic['large_large'].append(Article(4*i+3, 'largeTheta_largeBeta', large_theta, large_beta))


		return articlesDic['small_small'], 	articlesDic['small_large'], articlesDic['large_small'], articlesDic['large_large']



