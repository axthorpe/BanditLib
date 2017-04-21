import math
import numpy as np
import operator

class CascadeUCBUserStruct:
	def __init__(self, L):
		self.L = L
		self.w_hat = np.zeros(shape=L)
		self.T = np.ones(shape=L)
		self.U = []
		self.time = 1
		
	def c(self,t,s):
		return math.sqrt((1.5*math.log10(t))/s)
	
	def computeUCBs(self, E):
		Y = {}
		# compute UCBs
		if self.time == 1:
			for e in E:
				Y[e] = self.w_hat[e]
		else:
			for e in E:
				Y[e] = self.w_hat[e] + self.c(self.time-1, self.T[e])
		sortedU = sorted(Y.items(), key=operator.itemgetter(1), reverse=True)
		tempU = []
		for i in sortedU:
			tempU.append(i[0])
		self.U = tempU

	def updateParameters(self, A, C, K):
		for k in range(0, min(C+1,K)):
			e = A[k]
			self.T[e] = self.T[e] + 1
			self.w_hat[e] = ((self.T[e]-1)*self.w_hat[e] + (1 if C == k else 0))/(self.T[e]*1.0)
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
		


