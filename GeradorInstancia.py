from random import randint, choice
from abc import abstractmethod
from math import floor
from math import sqrt

def distanciaEuclidiana(x, y):
	x1, x2 = x
	y1, y2 = y
	return sqrt((y1 - x1) ** 2 + (y2 - x2) ** 2)

class GeradorInstanciaAbstrato ():
	@abstractmethod
	def gerar(self, n):
		pass

	def gerarGrafo(self, n):
		from graph_tool.all import complete_graph, graph_draw, sfdp_layout
		from numpy import array

		pontos, demandas, pis, qis = self.gerar(n)
		
		grafo = complete_graph(n, directed = True) # cria um digrafo completo

		pisProperty = grafo.new_vertex_property("int") # cria uma propriedade para guardar o nº de bicicletas inicial de cada vértice
		pisProperty.a = array(pis)

		qisProperty = grafo.new_vertex_property("int") # cria uma propriedade para guardar o nº alvo de bicicletas de cada vértice
		qisProperty.a = array(qis)

		pontosProperty = grafo.new_vertex_property("vector<float>"); # cria uma propriedade para guardas as coordenadas do ponto de cada vértice
		for i in range(n):
			pontosProperty[grafo.vertex(i)] = pontos[i]

		custoProperty = grafo.new_edge_property("float") # cria uma propriedade para guardar o custo de cada aresta
		aresta = None
		for e in grafo.edges():
			aresta = grafo.edge(e.target(), e.source())
			if not custoProperty[aresta]: 
				custoProperty[e] = distanciaEuclidiana(pontosProperty[e.source()], pontosProperty[e.target()])
			else:
				custoProperty[e] = custoProperty[aresta]

		grafo.vertex_properties["pi"] = pisProperty
		grafo.vertex_properties["qi"] = qisProperty
		grafo.edge_properties["custo"] = custoProperty
		grafo.vertex_properties["ponto"] = pontosProperty

		grafo.save("n{0}c{1}.xml".format(n, 1))
		
		graph_draw(grafo, pos = pontosProperty, vertex_text = grafo.vertex_index, vertex_font_size=18)
		
		

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
		demandas = [0] + [(randint(GeradorPrimeiraClasse.espacoDemanda[0], GeradorPrimeiraClasse.espacoDemanda[1])) for _ in range(n-1)]
		pis = [10 for _ in range(n)]
		qis = [10 + demandas[i] for i in range(n)]		

		return pontos, demandas, pis, qis

if __name__ == '__main__':
	GeradorPrimeiraClasse().gerarGrafo(5)