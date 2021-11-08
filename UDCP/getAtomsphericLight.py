import numpy as np

class Node(object):
	def __init__(self,x,y,value):
		self.x = x
		self.y = y
		self.value = value
	def printInfo(self):
		print(self.x,self.y,self.value)


def getAtomsphericLight(darkChannel, img):
    height = darkChannel.shape[0]
    width = darkChannel.shape[1]
    nodes = []
    for i in range(0, height):
        for j in range(0, width):
            oneNode = Node(i, j, darkChannel[i, j])
            nodes.append(oneNode)
    nodes = sorted(nodes, key=lambda node: node.value, reverse=True)

    atomsphericLight  = img[nodes[0].x, nodes[0].y, :]
    return atomsphericLight

