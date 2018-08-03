from random import randint, choice
from abc import abstractmethod
from math import floor

class GeradorInstanciaAbstrato ():
	@abstractmethod
	def gerar(self, n):
		pass

	def gerarGrafo(self, n):
		from graph_tool.all import complete_graph, graph_draw
		from numpy import array

		pontos, demandas, pis, qis = self.gerar(n)
		
		grafo = complete_graph(n, directed = True) # cria um digrafo completo
		
		pontosXProperty = grafo.new_vertex_property("int") #cria uma propriedade para guardar a coordenada x do ponto de cada vértice
		pontosXProperty.a = array(pontos)[:, 0]

		pontosYProperty = grafo.new_vertex_property("int") #cria uma propriedade para guardar a coordenada y do ponto de cada vértice
		pontosYProperty.a = array(pontos)[:, 1]

		pisProperty = grafo.new_vertex_property("int") # cria uma propriedade para guardar o nº de bicicletas inicial de cada vértice
		pisProperty.a = array(pis)

		qisProperty = grafo.new_vertex_property("int") # cria uma propriedade para guardar o nº alvo de bicicletas de cada vértice
		qisProperty.a = array(qis)

		graph_draw(grafo, vertex_text = grafo.vertex_index, vertex_font_size=18)
		

class GeradorSegundaClasse (GeradorInstanciaAbstrato):

	espacoPosicao = (0, 100)
	espacoDemanda = (1, 100)
	betas = (0., .05, .1, .2)
	
	def gerar(self, n):
		pontos = [(randint(GeradorSegundaClasse.espacoPosicao[0], GeradorSegundaClasse.espacoPosicao[1]),
				 	randint(GeradorSegundaClasse.espacoPosicao[0], GeradorSegundaClasse.espacoPosicao[1])) for _ in range(n)]
		demandas = [(randint(GeradorSegundaClasse.espacoDemanda[0], GeradorSegundaClasse.espacoDemanda[1])) for _ in range(n)]
		beta = choice(GeradorSegundaClasse.betas)
		f = lambda i : -1 if i % 2 == 1 else 1
		pis = [floor((1 - beta*f(i)) * demandas[i]) for i in range(n)]
		qis = [demandas[i] - pis[i] for i in range(n)]

		return pontos, demandas, pis, qis

class GeradorPrimeiraClasse(GeradorInstanciaAbstrato):
	
	espacoPosicao = (-500, 500)
	espacoDemanda = [-10, 10]

	def gerar(self, n):
		pontos = [(0, 0)] + [(randint(GeradorPrimeiraClasse.espacoPosicao[0], GeradorPrimeiraClasse.espacoPosicao[1]),
				 	randint(GeradorPrimeiraClasse.espacoPosicao[0], GeradorPrimeiraClasse.espacoPosicao[1])) for _ in range(n-1)]
		demandas = [(randint(GeradorPrimeiraClasse.espacoDemanda[0], GeradorPrimeiraClasse.espacoDemanda[1])) for _ in range(n)]
		pis = [10 for _ in range(n)]
		qis = [10 + demandas[i] for i in range(n)]		

		return pontos, demandas, pis, qis

if __name__ == '__main__':
	GeradorPrimeiraClasse().gerarGrafo(5)