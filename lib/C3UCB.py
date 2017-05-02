import math
import numpy as np
import operator

class CascadeUCBUserStruct:
	def __init__(self, delta, d):
		self.theta_hat = 0
		self.beta[delta] = 1
		self.V = np.identity(d)
		self.X = 
		self.Y = 
		self.time = 1
		
	def c(self,t,s):
		return math.sqrt((1.5*math.log10(t))/s)
	
	def computeUCBs(self, E, x):
		Y = {}
		x_norm = ???
		# compute UCBs
		tempU = np.zeros(len(E))
		for e in E:
			tempU[e] = np.transpose(self.theta_hat.dot(x) + self.beta[delta]*x_norm)

		sortedU = sorted(Y.items(), key=operator.itemgetter(1), reverse=True)
		tempU = []
		for i in sortedU:
			tempU.append(i[0])
		self.U = tempU

	def updateParameters(self, A, C, K):
		# for k in range(0, min(C+1,K)):
		# 	e = A[k]
		# 	self.T[e] = self.T[e] + 1
		# 	self.w_hat[e] = ((self.T[e]-1)*self.w_hat[e] + (1 if C == k else 0))/(self.T[e]*1.0)

		# EQUATIONS FOR UPDATE
		self.theta_hat = np.multiply(np.multiply(np.linalg.inv((np.multiply(((self.X).transpose), self.X) + np.identity(lambda1))), np.transpose(self.X)), self.Y)
		self.beta[delta] = R * math.sqrt(math.log(np.linalg.det(self.V)/((lambda1**d)*(delta*delta)))) + math.sqrt(lambda1)
		self.time += 1

	def getProb(self, element):
		return self.w_hat[element]
		
class CascadeUCBAlgorithm:
	def __init__(self, itemNum, n, K):  # n is number of users
		self.users = []
		#algorithm have n users, each user has a user structure
		for i in range(n):
			self.users.append(CascadeUCBUserStruct(itemNum)) 
		self.A = []
		self.itemNum = itemNum
		self.K = K
		self.CanEstimateUserPreference = False
		self.CanEstimateCoUserPreference = False
		self.CanEstimateW = False
		self.CanEstimateV = False

	def decide(self, pool_articles, userID):
		article_ids = []
		id_to_article = {}
		articles = []
		for article in pool_articles:
			article_ids.append(article.id)
			id_to_article[article.id] = article
		self.users[userID].computeUCBs(article_ids)
		self.A = (self.users[userID].U)[:self.K]
		for aid in self.A:
			articles.append(id_to_article[aid])
		return articles

	def updateParameters(self, click, userID):
		self.users[userID].updateParameters(self.A, click, self.K)
		


