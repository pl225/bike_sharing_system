from numpy.random import shuffle, choice, delete, vectorize, argmax, logical_not
from random import randint
from math import abs

"""
true false false true
0 1 2 3
-7 0 0 8

arg_where

pego os indices, shuffle os indices (d != 0)

pego os indices, tiro alguns e shuffle (d == 0)

"""

def computeTroca(d, Q, q): # demanda, Capacidade total, capacidade livre
	if d < 0:
		if abs(d) <= q:
			return abs(d)
		else:
			return q
	else:
		if Q - q >= d:
			return d
		else:
			return Q - q

class GeradorSolucaoAbstrato():

	cmtTroca = vectorize(computeTroca)

	def __init__(self, Q, demandas, vertices):
		self.capacidadeCaminhao = Q
		self.demandas = demandas
		mascara = self.demandas != 0
		self.verticesComDemanda = vertices[mascara]
		self.verticesSemDemanda = vertices[logical_not(mascara)]
		self.comDemandas = self.demandas[mascara]
		self.semDemandas = self.demandas[logical_not(mascara)]
		# criar dois numpy range, embaralhá-los e então usá-los como índices para demandas e vertices

	def algortimoConstrutivo(self):
		q = self.Q
		solucao = [0]
		shuffle(self.verticesComDemanda)
		verticesSemDemanda = []
		if self.verticesSemDemanda:
			verticesSemDemanda = choice(self.verticesSemDemanda, randint(0, len(self.verticesSemDemanda) - 1)
			shuffle(verticesSemDemanda, replace = False))
		OV = self.verticesComDemanda + verticesSemDemanda

		while OV:
			inserido = False
			i = 0
			while len(OV):
				if ((demandas[OV[i]] <= 0 and abs(demandas[OV[i]]) <= q) or (demandas[OV[i]] > 0 and self.Q - q >= demandas[OV[i]]): # mudança no artigo
					solucao.append(OV[i])
					q += demandas[OV[i]]
					inserido = True
					OV = delete(OV, i)
					break
				i += 1
			if not inserido:
				troca = cmtTroca(demandas, self.Q, q)
				indicesMaximos = argmax(troca)
				if len(indicesMaximos) > 1:
					pass # buscar mais próximo entre eles do vértice solucao[-1]
				else:
					solucao.append(indicesMaximos[0]) # mentira por enquanto. deveria ser OV[OV[troca[indicesMaximos[0]]]]
					demandas[troca[indicesMaximos[0]]] -= q # atualizar a demanda com q e atualizar q