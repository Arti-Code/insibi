#!/home/david/anaconda3/bin/python

# arguments
import argparse
from time import gmtime, strftime
parser = argparse.ArgumentParser(description='InSiBi')
parser.add_argument('-l', '--load', metavar='FILE', 	help='name of save or parameter file whith which to start')
parser.add_argument('-s', '--save', metavar='FILE', 	help='name for save prefix. default: \'./saves/<date>_<step>.dat\'')
parser.add_argument('-g', '--genes', metavar='FILE', 	help='prefix for genes file', default="genes")
parser.add_argument('-i', '--interval', metavar='INT',  help='interval in steps in which file is saved', type=int, default=10000)
parser.add_argument('-I', '--ignore', action='store_true',  help='ignore previous saved parameters, use InSiBi defaults')
parser.add_argument('-o', '--over', action='store_true',  help='overwrite previous saves')
parser.add_argument('-S', '--screensaver', action='store_true',  help='start in screensaver mode')
parser.add_argument('-r', '--resume', action='store_true',  help='load last save in date format from directory \'./saves/\'')
args = parser.parse_args()
if args.save is None: args.save = "saves/%s" % strftime("%Y-%m-%d-%H:%M", gmtime())
if args.load is not None and args.resume:
	print("Can't load and resume at the same time!")
	exit(1)

import sys
import random
import pymunk
from pymunk import Vec2d
import math
import numpy as np
import scipy
from scipy import ndimage
import cv2
import pickle
import select
from colour import Color
# from numba import jit
# random.seed(1)
# np.random.seed(1)

# parameters
## world
worldWidth	= 2000
worldHeight	= 1000
solventGrain 	= 50
# lightRate 	= 0.00003
lightRate 	= 0.01
sunSigmaX	= 7
sunSigmaY	= 7
sunMoveStep	= 5 # interval in which sun is moved
viscosity 	= 0.1 # fluid viscosity. Higher values means more viscous - faster objects break stronger
bondResRatio	= 1 # fluid resistance of bonds
minAttachment 	= 0.1
maxAttachment 	= 0.3
odorDegrade 	= 0.9
enzymeDegrade 	= 0.9
wasteDegrade 	= 0.9
dispersionRate	= 0.3
nSolvents 	= 7 # odors 1-3, energy/enzyme/waste, light
brown 		= 0
spawnSigma 	= 500 # 200
mutRange 	= 0.01
colMutRange 	= mutRange
UVmutateInterval = 5000 # UV event triggers more serious mutations than regular
UVmutRange	= 0.5
UVmutFrequ	= 0.001
UVcolMutRange 	= 0.1
## particle
maxAge 		= 1000
maxElasticity 	= 0.99
maxChlorophyl	= 1000
maxEnzyme	= 1000
maxTransporter	= 1000
divisionTime	= 5
geneInit	= 0.1
### bonds
maxBonds 	= 6
maxBondLen	= 200
bondStiff 	= 500
bondDamp 	= 100
bondRatio 	= 100		# activation to bond length
bondChangeRate	= 1		# change of length per timestep
bondEnRate 	= 10		# bond energy transfer rate per timestep
bondSolRate 	= 0.1		# relative amount of solvent transportable at one timestep
## cost and rate constants
maxEn 		= 3000
costExist 	= 1
costWaste 	= 1
costDivide 	= 500
rateTransfer 	= 50
costTransfer 	= 0.00001
## conversions
massToEn 	= 100
lightToEn	= 100
nutrientToEn	= 0.0001
enzymeToEn	= 0.0001
enToWaste 	= 0.0001
## costs for cytosol components
rateChlorophyl	= 0.00001
rateEnzyme	= 0.00001
rateTransporter = 0.00001
## effectiveness of cytosol components
rateHeterotrophy = 0.001
rateAutotrophy	= 0.001
rateExpression 	= 0.001

params = ["worldWidth", "worldHeight", "solventGrain", "lightRate", "sunSigmaX", "sunSigmaY", "viscosity", "bondResRatio", "minAttachment", "odorDegrade", "enzymeDegrade", "wasteDegrade", "dispersionRate", "nSolvents", "brown", "spawnSigma", "UVmutateInterval", "mutRange", "colMutRange", "UVmutateInterval", "UVmutRange", "UVmutFrequ", "UVcolMutRange", "maxAge", "maxElasticity", "maxChlorophyl", "maxEnzyme", "maxTransporter", "divisionTime", "maxBonds", "maxBondLen", "bondStiff", "bondDamp", "bondRatio", "bondChangeRate", "bondEnRate", "bondSolRate", "maxEn", "costExist", "costWaste", "costDivide", "rateTransfer", "costTransfer", "massToEn", "lightToEn", "nutrientToEn", "enzymeToEn", "enToWaste", "rateChlorophyl", "rateEnzyme", "rateTransporter", "rateHeterotrophy", "rateAutotrophy", "rateExpression", "sunX", "sunY"] 

class Particle(object):
	def check_remove(self):
		x = self.shape.body.position.x
		y = self.shape.body.position.y
		exitBorders = x < bX[0] or y < bY[0] or x > bX[1] or y > bY[1]
		noEnergy = self.energy <= 0
		oldAge = self.age > maxAge
		return exitBorders or noEnergy or oldAge
	def brownian(self):	
		if brown <= 0: return
		impulse = (random.randint(-brown,brown), random.randint(-brown,brown) )
		self.shape.body.apply_impulse_at_local_point(impulse)
	def transfer_energy(self, other, energy):
		if energy < 0: self, other = other, self
		demand = maxEn - self.energy
		energy = min(energy, demand) # self can't remove more energy than it can swallow
		supply = other.energy
		energy = min(energy, supply) # self can't remove more energy than there is in other
		other.energy -= energy
		self.energy += energy
		self.energy -= costTransfer
	def transfer_solutes(self, other, transfer):
		transfer = np.clip(transfer, 0, other.solvent[1:]) # clip to maximal the solvent concentration at the other cell
		self.solvent[1:] -= transfer
		other.solvent[1:] += transfer
	def move(self, x, y):
		p = self.shape.body.position
		p[0] += x
		p[1] += y
		self.shape.body.position = p
	def check_division(self): # dummy function
		return False
	def mutate(self):
		pass
	def draw_body(self):
		pass
	def draw_bonds(self):
		pass
	def ghostify(self):
		pass
	def reincarnate(self):
		pass
	def draw_information(self):
		pass

class Food(Particle):
	def __init__(self, x=0, y=0, energy=1000):
		self.age = 0
		self.energy = energy
		mass = self.energy / massToEn
		radius = max(1, radius_from_area(self.energy))
		inertia = pymunk.moment_for_circle(mass, 0, radius, (0,0))
		body = pymunk.Body(mass, inertia)
		body.position = x, y
		shape = pymunk.Circle(body, radius, (0,0))
		space.add(body, shape)
		self.shape = shape
		self.shape.Particle = self
		self.shape.collision_type = 2
	def draw_body(self):
		p = world_to_view(self.shape.body.position)
		pygame.draw.circle(screen, (255,255,255), p, max(2, int(self.shape.radius * Z)), 2)
	def step(self):
		self.age += 1
		self.energy -= costExist
		self._adjust_body()
		solvent = get_solvent(self.shape.body.position)
		solvent[4:7] = foodOdor
	def _adjust_body(self):
		mass = self.energy / massToEn
		radius = radius_from_area(self.energy)
		mass = max(1, mass)
		radius = max(1, radius)
		self.shape.unsafe_set_radius(radius)
		self.shape.body.mass = mass
	def draw_information(self):
		font = pygame.font.SysFont("monospace", 20)
		string = "E:%4d A:%4d" % (follow.energy, follow.age)
		text = font.render(string, 1, (255,255,255))
		textpos = text.get_rect()
		textpos.centerx = screen.get_rect().centerx
		screen.blit(text, textpos)

class Cell(Particle):
	def __init__(self, x=0, y=0, parrent=None):
		if parrent is None: 	# energy is needed for the following 
			self.energy = 		1000
		else:
			self.energy = 		parrent.energy	
		self.age = 0
		mass = max(1, self.energy / massToEn)
		self.mass = self.energy / maxEn
		radius = max(1, radius_from_area(self.energy))
		inertia = pymunk.moment_for_circle(mass, 0, radius, (0,0))
		body = pymunk.Body(mass, inertia)
		body.position = x, y
		shape = pymunk.Circle(body, radius, (0,0))
		space.add(body, shape)
		self.shape = shape
		self.shape.Particle = self
		self.shape.collision_type = 1
		self.solvent = get_solvent(self.shape.body.position)
		if parrent is None: 	# generate random genes and set interaction values to defaut values
			self.weights = 		np.random.normal(0, geneInit, (outputLength, stateLength))
			self.biases = 		np.random.normal(0, geneInit, (stateLength,1))
			self.states = 		np.zeros((stateLength, 1))
			self.outputs = 		np.zeros((outputLength,1))
			self.color = 		np.random.rand(3)
			self.odor = 		np.zeros(3)
			self.chlorophyl = 	1000
			self.enzyme = 		0
			self.transporter = 	1000
			self.enzymeExp = 	0
			self.enzymeImob = 	0
			self.bonding = 		1
			self.oszillFreq = 	0
			self.bLength = 		1
			self.bStiff = 		1
			self.bDamp = 		1
			self.bEnTrans = 	0
			self.bInfoRead = 	np.array([0.0,0.0,0.0])
			self.bInfoWrite = 	np.array([0.0,0.0,0.0])
			self.bSolTrans = 	np.zeros(6)
		else: 			# copy genes and interaction values
			self.energy = 		parrent.energy	
			self.shape.body.velocity = parrent.shape.body.velocity
			self.weights = 		np.copy(parrent.weights)
			self.biases = 		np.copy(parrent.biases)
			self.states = 		np.copy(parrent.states)
			self.outputs = 		np.copy(parrent.outputs)
			self.color = 		np.copy(parrent.color)
			self.odor = 		np.copy(parrent.odor)
			self.chlorophyl = 	parrent.chlorophyl	
			self.enzyme = 		parrent.enzyme
			self.transporter =	parrent.transporter	
			self.enzymeExp = 	parrent.enzymeExp
			self.enzymeImob = 	parrent.enzymeImob
			self.bonding = 		parrent.bonding
			self.oszillFreq = 	parrent.oszillFreq
			self.bLength = 		parrent.bLength
			self.bStiff = 		parrent.bStiff
			self.bDamp = 		parrent.bDamp
			self.bEnTrans = 	parrent.bEnTrans
			self.bInfoRead =	np.copy(parrent.bInfoRead)
			self.bInfoWrite = 	np.copy(parrent.bInfoWrite)
			self.bSolTrans = 	np.copy(parrent.bSolTrans)
	def step(self):
		# initial checkups
		self.solvent = get_solvent(self.shape.body.position)
		self.mass = self.energy / maxEn
		light = self.solvent[0]
		nutrient = self.solvent[1]
		enzyme = self.solvent[2]
		waste = self.solvent[3]
		## autophagy
		harvestedLight = max(0, min(1, light * self.chlorophyl * rateAutotrophy))# / self.mass)
		self.solvent[0] -= harvestedLight
		self.energy += lightToEn * harvestedLight
		self.energy -= rateChlorophyl * self.chlorophyl
		## heterotrophy
		harvestedNutrient = max(0, min(1, nutrient * self.transporter * rateHeterotrophy))# / self.mass)
		self.solvent[1] -= harvestedNutrient
		self.energy += nutrientToEn * harvestedNutrient
		self.energy -= rateTransporter * self.transporter
		## enzyme expression
		producedEnzyme = max(0, min(1, self.enzyme * rateExpression * self.enzymeExp))# / self.mass)
		self.solvent[2] += producedEnzyme
		self.energy -= rateEnzyme * self.enzyme
		## general costs and solvent depositions
		self.energy -= costExist
		self.energy = max(0, min(self.energy, maxEn))
		self.energy -=  costWaste * self.mass * self.solvent[3]
		lostEnergy = min(self.energy, enzyme * rateEnzyme) # cell could expell more nutrients than possible without min()
		self.energy -= lostEnergy
		lostNutrient = lostEnergy / nutrientToEn
		self.solvent[1] += lostNutrient # add nutrient to solvent
		self.solvent[3] += enToWaste * self.energy
		self.solvent[4:7] += self.odor
		# regulatory network
		## input
		self.states[0,0] 	= self.energy / maxEn 				# IN energy
		self.states[1:8,0] 	= self.solvent					# IN light, free energy, enzyme, waste, odor1-3
		self.states[8,0] 	= len(self.shape.body.constraints) / maxBonds	# IN number bonds
		self.states[9:12,0] 	= self.bInfoRead				# IN information recieved over bonds
		self.states[12,0] 	= self.chlorophyl / maxChlorophyl 		# IN chlorophyl
		self.states[13,0] 	= self.enzyme / maxEnzyme			# IN enzyme
		self.states[14,0] 	= self.transporter / maxTransporter 		# IN transporter
		self.states[15,0] 	= math.cos(step*self.oszillFreq)		# IN oszillator
		self.states[:nInputs,0] = self.states[:nInputs,0] * 2 - 1		# zero center
		## nonlinearity
		self.states[nInputs:,] 	= np.tanh( np.dot(self.weights, self.states + self.biases )) # nonlinearity
		self.states 		= np.clip(self.states, -1, 1) 			# clip activations 
		self.outputs 		= self.states[-nOutputs:,]			# make outputs
		self.outputs 		= np.clip(self.outputs, -1, 1) 			# clip outputs
		self.outputs 		= self.outputs * 0.5 + 0.5			# get to the range 0-1
		## output
		self.division 		= self.outputs[0,0] 				# OUT division
		self.odor 		= self.outputs[1:4,0]				# OUT odor
		self.elasticity 	= self.outputs[4,0]		 		# OUT elasticity
		self.bonding 		= self.outputs[5,0]				# OUT bonding propensity
		self.bLength 		= self.outputs[6,0]				# OUT bond length
		self.attachment 	= self.outputs[7,0]				# OUT attachment to surface
# 		self.attachment 	= 0
		growChlorophyl 		= self.outputs[8,0]				# OUT clorophyl growing
		growEnzyme 		= self.outputs[9,0]				# OUT clorophyl growing
		growTransporter 	= self.outputs[10,0]				# OUT clorophyl growing
		self.bInfoWrite		= self.outputs[11:14,0]				# OUT information sent over bonds
		self.bEnTrans 		= self.outputs[14,0]				# OUT energy sent over bonds
		self.bSolTrans 		= self.outputs[15:21,0]				# OUT solvent and odor transferred over bonds
		self.enzymeExp 		= self.outputs[21,0]				# OUT enzyme expression rate
		self.enzymeImob 	= self.outputs[22,0]				# OUT enzyme surface imobilisation
		self.oszillFreq 	= self.outputs[23,0]				# OUT enzyme surface imobilisation
		# adjust body and bond attributes, deposit substances
		self.age += 1
		self._adjust_body()
		self._adjust_bonds()
		self._adjust_forces()
		self.bInfoRead.fill(0) # reset bond info buffer
		if self.chlorophyl < maxChlorophyl:
			self.chlorophyl += growChlorophyl
		if self.enzyme < maxEnzyme:
			self.enzyme += growEnzyme
		if self.transporter < maxTransporter:
			self.transporter += growTransporter
	def mutate(self):
		# weights
		mutations = np.random.normal(0, mutRange, self.weights.shape)
		self.weights += mutations
		self.weights = np.clip(self.weights, -maxWeight, maxWeight)
		# biases
		mutations = np.random.normal(0, mutRange, self.biases.shape)
		self.biases += mutations
		self.biases = np.clip(self.biases, -maxWeight, maxWeight)
		# color
		mutations = np.random.normal(0, colMutRange, self.color.shape)
		self.color += mutations
		self.color = np.clip(self.color, 0, 1)
	def mutate_UV(self):	
		# weights
		mutations = np.random.normal(0, UVmutRange, self.weights.shape)
		pick = np.random.choice([True, False], size=mutations.shape, p=[UVmutFrequ, 1-UVmutFrequ])
		self.weights = np.where(pick, self.weights+mutations, self.weights) # where false, keep values, otherwhise replace with mutated
		self.weights = np.clip(self.weights, -maxWeight, maxWeight)
		# biases
		mutations = np.random.normal(0, UVmutRange, self.biases.shape)
		pick = np.random.choice([True, False], size=mutations.shape, p=[UVmutFrequ, 1-UVmutFrequ])
		self.biases = np.where(pick, self.biases+mutations, self.biases) # where false, keep values, otherwhise replace with mutated
		self.biases = np.clip(self.biases, -maxWeight, maxWeight)
		# color
		mutations = np.random.normal(0, UVcolMutRange, self.color.shape)
		self.color += mutations
		self.color = np.clip(self.color, 0, 1)
	def check_division(self):
		if (self.energy - costDivide) / 2 < 0: return False
		return self.division > 0
	def divide(self):
		self.energy -= costDivide
		self.energy /= 2
		self.chlorophyl /= 2
		self.enzyme /= 2
		self.transporter /= 2
		self._adjust_body()
		self.age = 0
		x, y = self.shape.body.position
		x += random.uniform(-1e-8, 1e-8) # add a very small difference in position, to deter budding in one distinct direction
		y += random.uniform(-1e-8, 1e-8)
		newCell = Cell(x, y, self)
		newCell.mutate()
		bonds = list(self.shape.body.constraints)
		bonds.sort(key=lambda x: -x.rest_length) # sort for shortest distances
		bonds = bonds[:5]
		for bond in bonds: # bond to all the cells parrent is bound to.
			cellA = bond.cellA
			cellB = bond.cellB
			if cellA is self: 	other = cellB
			else:			other = cellA
			if other.check_remove(): continue
			newCell.bond(other)
		self.bond(newCell)
		return newCell
	def draw_body(self):
		p = world_to_view(self.shape.body.position)
		color = (int(self.color[0]*255), int(self.color[1]*255), int(self.color[2]*255))
		pygame.draw.circle(screen, color, p, max(2, int(self.shape.radius * Z)))
	def draw_cytosol(self):
		p = world_to_view(self.shape.body.position)
		colorMax = max([self.enzyme/maxEnzyme, self.chlorophyl/maxChlorophyl, self.transporter/maxTransporter])
		color = (int(self.enzyme/maxEnzyme/colorMax*255), int(self.chlorophyl/maxChlorophyl/colorMax*255), int(self.transporter/maxTransporter/colorMax*255))
		pygame.draw.circle(screen, color, p, max(2, int(self.shape.radius * Z)))
	def draw_bonds(self):
		for bond in self.shape.body.constraints:
			cellA = bond.cellA
			cellB = bond.cellB
			p1 = cellA.shape.body.position
			p2 = cellB.shape.body.position
			p1 = world_to_view(p1)
			p2 = world_to_view(p2)
			bInfo = (cellA.bInfoWrite + cellB.bInfoWrite) / 2
			red = 	int(min(1, max(0.3, bInfo[0]/2 + 0.5)) * 255)
			blue = 	int(min(1, max(0.3, bInfo[1]/2 + 0.5)) * 255)
			green = int(min(1, max(0.3, bInfo[2]/2 + 0.5)) * 255)
			pygame.draw.line(screen, (red,blue,green), p1, p2, 2)
	def bond(self, other):
		if self.bonding < 0 or other.bonding < 0: return
		if len(self.shape.body.constraints) >= maxBonds or len(other.shape.body.constraints) >= maxBonds: return
		distance = self.shape.body.position.get_distance( other.shape.body.position )
		length = self.shape.radius + other.shape.radius + distance # bond lengths are between surfaces of cells
		spring = pymunk.constraint.DampedSpring(self.shape.body, other.shape.body, (0,0), (0,0), length, bondStiff, bondDamp)
		if self.age == 0: spring.collide_bodies = False # adjecent bodies don't collide -> realistic division behaviour
		space.add(spring)
		spring.cellA = self
		spring.cellB = other
	def ghostify_bonds(self):
		self.ghostBonds = []
		for bond in self.shape.body.constraints:
			if bond.cellB == self: continue # only let cellA ghostify bonds
			self.ghostBonds.append(bond.cellB)
	def ghostify(self):
		self.ghostPosition = self.shape.body.position
		self.ghostVelocity = self.shape.body.velocity
		space.remove(self.shape.body, self.shape)
		self.shape.body = None
		self.shape = None
	def reincarnate(self):
		mass = max(1, self.energy / massToEn)
		self.mass = self.energy / maxEn
		radius = max(1, radius_from_area(self.energy))
		inertia = pymunk.moment_for_circle(mass, 0, radius, (0,0))
		body = pymunk.Body(mass, inertia)
		body.position = self.ghostPosition
		shape = pymunk.Circle(body, radius, (0,0))
		space.add(body, shape)
		self.shape = shape
		self.shape.Particle = self
		self.shape.collision_type = 1
		self.shape.body.velocity = self.ghostVelocity
		self.solvent = get_solvent(self.shape.body.position)
	def reincarnate_bonds(self):
		for bond in self.ghostBonds:
			self.bond(bond)
	def draw_information(self):
		font = pygame.font.SysFont("monospace", 20)
		string = "E:%4d A:%4d Chl:%3d Enz:%3d Tra:%3d" % (follow.energy, follow.age, follow.chlorophyl, follow.enzyme, follow.transporter)
		text = font.render(string, 1, (255,255,255))
		textpos = text.get_rect()
		textpos.centerx = screen.get_rect().centerx
		screen.blit(text, textpos)

		subWidth = screenOffX - 25
		subHeight = screenOffY*2 - 100
		subX = screenOffX
		subY = 25
		sub = pygame.Surface((subWidth, subHeight))
		sub.set_alpha(200)
		sub.fill((255,255,255))
			
		xSpaceProts1 = int(subWidth * 0.03)
		xSpaceProts2 = int(subWidth * 0.30)
		xSpaceProts3 = int(subWidth * 0.73)
		xSpaceProts4 = int(subWidth * 0.75)
		ySpaceInputs = int((subHeight-50) / stateLength)
		ySpaceOutputs = int((subHeight-50) / outputLength)

		displayStates = self.states / 2 + 0.5
		if not hasattr(self, "prevStates"): # check to see if previous state is logged
			self.prevStates = self.states
			self.prevDispStates = displayStates

		for i in range(stateLength):
			p1 = (xSpaceProts2, ySpaceInputs*i + 25)
			for j in range(outputLength):
				p2 = (xSpaceProts3, ySpaceOutputs*j + 25)
				n = min(1, max(0, float(self.weights[j,i]*0.5+0.5)))
				color = colorScale[int(n*255)]
				color = (int(color.red*255), int(color.green*255), int(color.blue*255))
				pygame.draw.lines(sub, color, False, [p1, p2], 1)

			p = (xSpaceProts1, ySpaceInputs*i + 15)
			textPro = font.render("%s % 4.2f" % (stateNames[i].ljust(12), self.prevStates[i]), 1, (50,50,50))
			sub.blit(textPro, p)

			p = (xSpaceProts2, ySpaceInputs*i + 25)
			n = min(1, max(0, float(self.prevDispStates[i])))
			color = colorScale[int(n*255)]
			color = (int(color.red*255), int(color.green*255), int(color.blue*255))
			pygame.draw.circle(sub, color, p, 10)

		for i in range(outputLength):
			p = (xSpaceProts3, ySpaceOutputs*i + 25)
			n = min(1, max(0, float(self.biases[i]*0.5+0.5)))
			color = colorScale[int(n*255)]
			color = (int(color.red*255), int(color.green*255), int(color.blue*255))
			pygame.draw.circle(sub, color, p, 15)
			n = min(1, max(0, float(displayStates[i+nInputs])))
			color = colorScale[int(n*255)]
			color = (int(color.red*255), int(color.green*255), int(color.blue*255))
			pygame.draw.circle(sub, color, p, 10)

			p = (xSpaceProts4, ySpaceOutputs*i + 15)
			textPro = font.render("% 4.2f %s" % (self.states[i+nInputs], stateNames[i+nInputs]), 1, (50,50,50))
			sub.blit(textPro, p)

		screen.blit(sub, (subX, subY))
		self.prevStates = self.states
		self.prevDispStates = displayStates

	def _adjust_body(self):
		mass = self.energy / massToEn # mass
		mass = max(1, mass)
		self.shape.body.mass = mass
		radius = radius_from_area(self.energy) # radius
		radius = max(1, radius)
		self.shape.unsafe_set_radius(radius)
		self.shape.elasticity = min(maxElasticity, self.elasticity)
	def _adjust_bonds(self):
		bondsToRemove = []
		for bond in self.shape.body.constraints:
			lead = True # only set length once
			cellA = bond.cellA
			cellB = bond.cellB
			if cellA == self:
				other = cellB
			else:
				other = cellA
				lead = False
			if other.check_remove(): # if other cell is dead, ignore bond and delete it
				bond.cellA = None # unset references to the dead cell, otherwhise the dead cell will point to the bond and the bond
				bond.cellB = None #  to the cell, making it impossible to remove either
				space.remove(bond)
				continue
			other.bInfoRead += self.bInfoWrite
			self.bInfoRead += other.bInfoRead
			self.transfer_energy(other, self.bEnTrans * bondEnRate)
			self.transfer_solutes(other, self.bSolTrans * bondSolRate)
			if lead:
				if ( self.age > divisionTime and self.bonding < 0) or ( other.age > divisionTime and other.bonding < 0):
					space.remove(bond)
					continue
				if self.age > divisionTime:
					bond.collide_bodies = False # adjecent bodies don't collide -> realistic division behaviour
				targetLength = self.mass*self.bLength*bondRatio + other.mass*other.bLength*bondRatio + self.shape.radius + other.shape.radius # bond lengths are only between surfaces of cells
				bond.rest_length += (targetLength - bond.rest_length) * bondChangeRate 
				length = (self.shape.body.position - other.shape.body.position).length - (self.shape.radius + other.shape.radius)
				if length > maxBondLen:
					space.remove(bond)
					continue
	def _adjust_forces(self):
		velocVec = self.shape.body.velocity
		if velocVec.length == 0: return
		# fluid resistance of bonds
		dragVec = pymunk.vec2d.Vec2d(0, 0)
		for bond in self.shape.body.constraints: # go through all bonds
			posA = bond.cellA.shape.body.position
			posB = bond.cellB.shape.body.position
			bondVec = posA - posB
			if bondVec.length == 0: continue
			bondVec.angle_degrees += 90 # get normal vecor to bond
			angle = velocVec.get_angle_degrees_between(bondVec)
			if angle > 90 or angle < -90: # rotate if necessary
			       dragVec += bondVec
			else:
			       dragVec -= bondVec
		dragVec *= bondResRatio
		self.shape.body.apply_force_at_local_point(dragVec,(0,0))
		# fluid resistance of body
		viscRatio = float(e ** (- velocVec.get_length()*viscosity*max(minAttachment, min( maxAttachment, self.attachment)))) 
		self.shape.body.velocity *= viscRatio

def radius_from_area(area):
	if area <= 0: return 0
	return math.sqrt(area / 3.141592)
def world_to_view(p):
	x, y = p
	return int((x + X) * Z + screenOffX ), int((y + Y) * Z + screenOffY)
def view_to_world(p):
	x, y = p
	return int(((x - screenOffX) / Z ) - X ), int(((y - screenOffY) / Z ) - Y )
def add_borders(space):
	body = pymunk.Body(body_type = pymunk.Body.STATIC)
	body.position = (0, 0)
	l1 = pymunk.Segment(body, (bX[0], bY[0]),  (bX[0], bY[1]), 5)
	l2 = pymunk.Segment(body, (bX[0], bY[0]),  (bX[1], bY[0]), 5)
	l3 = pymunk.Segment(body, (bX[1], bY[1]),  (bX[0], bY[1]), 5)
	l4 = pymunk.Segment(body, (bX[1], bY[1]),  (bX[1], bY[0]), 5)
	space.add(l1, l2, l3, l4)
	return l1,l2, l3, l4
def draw_lines(screen, lines):
	for line in lines:
		body = line.body
		pv1 = body.position + line.a.rotated(body.angle)
		pv2 = body.position + line.b.rotated(body.angle)
		p1 = world_to_view(pv1)
		p2 = world_to_view(pv2)
		pygame.draw.lines(screen, (255,255,255), False, [p1,p2])
def collision_cell_with_food(arbiter, space, data):
	a,b = arbiter.shapes
	cell = a.Particle
	food = b.Particle
	cell.transfer_energy(food, -rateTransfer)
def collision_cell_with_cell(arbiter, space, data):
	a,b = arbiter.shapes
	cellA = a.Particle
	cellB = b.Particle
	cellA.bond(cellB)
	enzyme = cellA.enzymeImob - cellB.enzymeImob
	cellA.transfer_energy(cellB, -enzyme*enzymeToEn)
def disperse_solvent(solvent):
	return solvent +  (cv2.blur(solvent, (3,3)) - solvent) * dispersionRate # dirty hack to make smoothing slower
def get_solvent(pos):
	x, y = pos
	x = x + wX / 2 # x with 0 at leftmost corner of world
	i = int(x/solventGrain)
	y = y + wY / 2 # y with 0 at leftmost corner of world
	j = int(y/solventGrain)
	return solvent[i,j,:]
def draw_solvent(light=False, solute=False, odor=False):
	drawN = int(light) + int(solute) + int(odor)
	if drawN == 0: return
	color = np.zeros(3)
	x1, y1 = view_to_world( (0, 0) ) # get points for rectangle of world coordinates displayed on screen
	x2, y2 = view_to_world( (screenOffX*2,screenOffY*2) )
	x1 = x1 + wX/2 # de-center world coordinates, so that top-right of map is 0,0
	y1 = y1 + wY/2
	x2 = x2 + wX/2
	y2 = y2 + wY/2
	i1 = max(0, min(solventNx, int(x1/solventGrain) )) # get indices for solvent
	j1 = max(0, min(solventNy, int(y1/solventGrain) ))
	i2 = max(0, min(solventNx, int(x2/solventGrain) ))
	j2 = max(0, min(solventNy, int(y2/solventGrain) ))
	recXwidth = int( (screenOffX*2) / (i2-i1) )
	recYwidth = int( (screenOffY*2) / (j2-j1) )
	for i in range(i1,i2):
		for j in range(j1, j2):
			color.fill(0)
			if light:
				light = solvent[i,j,0]
				color += (255*light)/drawN
			if solute:
				nutrient = solvent[i,j,1]
				enzyme = solvent[i,j,2]
				waste = solvent[i,j,3]
				color[0] += (255*enzyme)/drawN
				color[1] += (255*waste)/drawN
				color[2] += (255*nutrient)/drawN
			if odor:
				odor1 = solvent[i,j,4]
				odor2 = solvent[i,j,5]
				odor3 = solvent[i,j,6]
				color[0] += (255*odor1)/drawN
				color[1] += (255*odor2)/drawN
				color[2] += (255*odor3)/drawN
			x = int( (i-i1) * recXwidth )
			y = int( (j-j1) * recYwidth )
			pygame.draw.rect(screen, color, (x,y,recXwidth, recYwidth))
def get_parameter_string():
	paramDict = {name : eval(name) for name in params}
	string = ""
	for param in params:
		paramVal = paramDict[param]
		paramString = param + ' '*(20-len(param)) # padd parameter names with spaces
		if type(paramVal) is int: # differentiate between ints and floats
			string += "# %s %d\n" % (paramString, paramDict[param])
		elif type(paramVal) is float:
			string += "# %s %f\n" % (paramString, paramDict[param])
	return string
def str_to_parameter(string):
	lines = string.split('\n')
	for line in lines:
		if line == '' or line[0] != '#': continue # ignore empty or non-comment lines
		param, value = line[2:].split() # ignore comment 
		try: # set global values, as int or float
			globals()[param] = int(value)
		except ValueError:
			globals()[param] = float(value)
def sun_move(x,y,sunpatch):
	x = solventNx + x
	y = solventNy + y
	iFromX = list(range(solventNy*2))
	iFromY = list(range(solventNy*2))
	iToX = iFromX[x:(solventNx*2)] + iFromX[0:x]
	iToY = iFromY[y:(solventNy*2)] + iFromY[0:y]
	tmpSun = np.zeros_like(sun)
	tmpSun[iToX,:] = sun[iFromX,:]
	tmpSun[:,iToY] = sun[:,iFromY]
	sunpatch = np.copy(tmpSun[solventNx//2:solventNx+solventNx//2 ,solventNy//2:solventNy+solventNy//2])

	

# constants
## particles
foodOdor 	= np.array([0,0,0.5])
maxWeight	= 1
maxActivation 	= 1
## genes
inputNames = ["energy", "light", "nutrient", "enzyme", "waste", "odorR", "odorG", "odorB", "nBonds", "bondInfoR", "bondInfoG", "bondInfoB", "chlorophyll", "expression", "transporter", "oszillator"]
internalNames = ["int1", "int2", "int3", "int4", "int5", "int6", "int7", "int8", "int9", "int10", "int11", "int12", "int13", "int14", "int15", "int16", "int17", "int18", "int19", "int20"]
outputNames = ["division", "elasticity", "odorR", "odorG", "odorB", "bonding", "bondLen", "attachment", "bondInfoR", "bondInfoG", "bondInfoB", "bondEnTran", "transNut", "transEnz", "transWaste", "trasOdorR", "transOdorG", "transOdorB", "growChloro", "growExpr", "growTrans", "enzymeEx", "enzymeImmob", "oszillFrequ"]
stateNames 	= inputNames + internalNames + outputNames
nInputs 	= len(inputNames)
nInternals 	= len(internalNames)
nOutputs 	= len(outputNames)
inputLength  	= nInputs + nInternals 
stateLength  	= nInputs + nInternals + nOutputs # values for inputs, internal states and outputs are also aranged like that calculation time
outputLength 	=           nInternals + nOutputs
## math
e		= 2.71828182845 # Euler's number
pi		= 3.14159265359 # Pi
## derived constants
bX 		= (-worldWidth, worldWidth)
bY 		= (-worldHeight, worldHeight)
wX 		= bX[1] - bX[0]
wY 		= bY[1] - bY[0]
solventNx 	= int( (bX[1] - bX[0]) / solventGrain )
solventNy 	= int( (bY[1] - bY[0]) / solventGrain )
sunXgaus 	= cv2.getGaussianKernel(solventNx*2, sunSigmaX)
sunYgaus 	= cv2.getGaussianKernel(solventNy*2, sunSigmaY)
sun 		= sunXgaus.dot(sunYgaus.T)
sunX		= solventNx 
sunY		= solventNy
centerInt 	= sun[solventNx, solventNy]
sun 		= sun / centerInt # make center value 1
colorScale	= list(Color("red").range_to(Color("white"), 128)) + list(Color("white").range_to(Color("blue"), 128)) # color scale from red to white to blue
# global data structures
space = pymunk.Space(threaded=True)
space.threads = 2 # max number of threads
space.iterations = 1 # minimum number of iterations for rough but fast collision detection
lines = add_borders(space)
solvent = np.zeros( (solventNx, solventNy, nSolvents) )
# solvent[:,:,0] = np.full( (solventNx, solventNy), 0.2)
# view behaviour
X, Y = 0, 0
Z = 0.45
# world behaviour
hadler_CF = space.add_collision_handler(1, 2)
hadler_CF.post_solve = collision_cell_with_food
hadler_CC = space.add_collision_handler(1, 1)
hadler_CC.post_solve = collision_cell_with_cell
particles = []
particlesToRemove = []
particlesToAdd = []
running = True
pause = False
display = False
seed = False
food = False
save = False
saveGenes = False
step = 1

if args.screensaver:
	import pygame
	from pygame.locals import *
	import pymunk.pygame_util
	pygame.init()
	infoObject = pygame.display.Info()
	dim = (infoObject.current_w, infoObject.current_h)
	screenOffX, screenOffY = infoObject.current_w / 2, infoObject.current_h / 2
	screen = pygame.display.set_mode(dim)
	clock = pygame.time.Clock()
	display = True
	displayLight = False
	displaySolute = False
	displayOdor = False
	displayCytosol = False
	follow = None
	pygame.display.toggle_fullscreen()

# load file
if args.load is not None:
	loadFile = open(args.load, "rb")
if args.resume:
	from os import listdir
	from os.path import isfile, join
	import re
	files = [f for f in listdir('saves') if isfile(join('saves', f))]
	files = [file for file in files if re.match("[0-9]{4}-[0-9]{2}-[0-9]{2}-[0-9]{2}:[0-9]{2}_[0-9]*.dat", file)]
	if len(files) == 0:
		print("No recent saves to resume.")
		exit(0)
	files.sort()
	loadFileName = "./saves/" + files[-1]
	loadFile = open(loadFileName, "rb")

if args.load is not None or args.resume:
		string = ""
		filePointer = None # file pointer to go to after reading parameters
		for line in loadFile:
			try: # deconding will fail at some point
				string += line.decode('utf-8')
			except:
				break
			filePointer = loadFile.tell()
		if not args.ignore:
			str_to_parameter(string)
		loadFile.seek(filePointer)
		try:
			step = pickle.load(loadFile)
			particles = pickle.load(loadFile)
			for particle in particles:
				particle.reincarnate()
			for particle in particles:
				particle.reincarnate_bonds()
			solvent = pickle.load(loadFile)
			print("Read save file including %d cells at step %d." % (len(particles), step))
		except EOFError:
			print("Read parameters file.")
		
# main loop
while running:
	if not pause:
		print("step:%8d" % (step), end="\r")
		if step % 100 == 0:
			bonds = [len(list(cell.shape.body.constraints)) for cell in particles]
			bondR = sum(bonds)/max(1, float(len(particles))) # guard against division by zero
			print("step:%8d\tcells:%5d\tbond-ratio: %2.2f" % (step, len(particles), bondR), end="\r")
		if step % 1000 == 0:
			weights = []
			for particle in particles:
				weights.append(particle.weights)
			weights = np.array(weights)
			weightsStd = np.sum(np.std(weights, axis=0)) / (weights.shape[0]*weights.shape[1]) # standard deviation of weights, has to be normalized by number weights
			print("step:%8d\tcells:%5d\tbond-ratio: %2.2f\tgene_std: %f" % (step, len(particles), bondR, weightsStd))

		if len(particles) == 0 or seed:
			seed = False
			step = 1
			for i in range(50):
				x = random.randint(bX[0]+1, bX[1]-1)	
				y = random.randint(bY[0]+1, bY[1]-1)	
				particles.append(Cell(x=x, y=y))
		if food:
			food = False
			for i in range(100):
				x = random.gauss(0, spawnSigma)	
				y = random.gauss(0, spawnSigma)	
				particles.append(Food(x=x, y=y))
			
		particlesToRemove = []
		for particle in particles: 
			if particle.check_remove(): 	particlesToRemove.append(particle)
		for particle in particlesToRemove:
			space.remove(particle.shape.body, particle.shape)
			particles.remove(particle)

		particlesToRemove = []
		for particle in particles: 
			particle.brownian()
			particle.step()
			if particle.check_division(): 	particlesToAdd.append(particle.divide())
			if particle.check_remove(): 	particlesToRemove.append(particle)

		for particle in particlesToRemove:
			space.remove(particle.shape.body, particle.shape)
			particles.remove(particle)
		particles.extend(particlesToAdd)
		particlesToAdd = []

		if step % UVmutateInterval == 0:
			for particle in particles: particle.mutate_UV()
			print("UV event")

		# solvent and light
		## sun movement
		if step % sunMoveStep == 0:
			case = random.randint(0,3)
			if sunX < 3*solventNx//2 and case == 0: sunX += 1
			if sunX > solventNx//2 and case == 1: sunX -= 1
			if sunY < 3*solventNy//2 and case == 2: sunY += 1
			if sunY > solventNy//2 and case == 3: sunY -= 1
		## solvent
		solvent[:,:,0] += 	lightRate * sun[(sunX-solventNx//2):(sunX+solventNx//2), (sunY-solventNy//2):(sunY+solventNy//2)]
		solvent[:,:,2] *= 	enzymeDegrade
		solvent[:,:,3] *= 	wasteDegrade
		solvent[:,:,4:7] *= 	odorDegrade
		solvent = disperse_solvent(solvent)
		solvent = np.clip(solvent, 0, 1)

		space.step(1/50.0)
		step += 1
		if display:
			clock.tick(50)
			
		if save or step%args.interval==0 and not len(particles)==0:
			save = False
			if args.over: 	saveName = "%s.dat" % (args.save)
			else:		saveName = "%s_%06d.dat" % (args.save, step)
			with open(saveName, "wb") as file:
				paramString = get_parameter_string()
				file.write(str.encode(paramString)) # encode string as binary
				cells = []
				for particle in particles:
					if isinstance(particle, Cell):
						cells.append(particle)
				for cell in cells:
					cell.ghostify_bonds()
				for cell in cells:
					cell.ghostify()
				pickle.dump(step, file)
				pickle.dump(cells, file)
				pickle.dump(solvent, file)
				for cell in cells:
					cell.reincarnate()
				for cell in cells:
					cell.reincarnate_bonds()
			print("\nsaved file %r" % saveName)
		
		if saveGenes:
			saveGenes = False
			fileName = "%s_%d.dat" % (args.genes, step)
			with open(fileName, "w") as file:
				for particle in particles:
					for row in particle.weights:
						for weight in row: 
							file.write("% 4.2f " % weight)				
						file.write("\n")
					for bias in particle.biases:			
						file.write("% 4.2f " % bias)
					file.write("\n\n")


# input
	inStr = select.select([sys.stdin,],[],[],0.0)[0] # check if input is pending
	if inStr:
		inStr = input()
		if inStr == "":
			pause = not pause
		elif inStr == "display" or inStr == "d":
			import pygame
			from pygame.locals import *
			import pymunk.pygame_util
			pygame.init()
			infoObject = pygame.display.Info()
			dim = (infoObject.current_w, infoObject.current_h)
			screenOffX, screenOffY = infoObject.current_w / 2, infoObject.current_h / 2
			screen = pygame.display.set_mode(dim)
			clock = pygame.time.Clock()
			display = True
			displayLight = False
			displaySolute = False
			displayOdor = False
			displayCytosol = False
			follow = None
			fullcreen = False
		elif inStr == "save" or inStr == "s":
			save = True
		elif inStr == "genes" or inStr == "g":
			saveGenes = True
		elif inStr == "exit" or inStr == "e":
			running = False

# graphic output 
	if display:
		pygame.display.set_caption("InSiBi\tFPS: %2.2f\tStep: %8d\t[%d:%d] zoom: %2.2f\tCells: %3d\tSun: %4d,%4d" % (clock.get_fps(), step, X, Y, Z, len(particles), sunX - solventNx//2, sunY - solventNy//2))
		screen.fill((0,0,0))
		draw_lines(screen, lines)
		draw_solvent(displayLight, displaySolute, displayOdor)
		for particle in particles:
			particle.draw_bonds()
		for particle in particles:
			particle.draw_body()
			if displayCytosol:
				particle.draw_cytosol()
		if follow:
			X, Y = - follow.shape.body.position
			X -= screenOffY / Z / 2
			follow.draw_information()
			if follow.check_remove():
				follow = None
		pygame.display.flip()
		keys = pygame.key.get_pressed()
		if keys[K_RIGHT]:
			X -= 10
			follow = None
		if keys[K_LEFT]:
			X += 10
			follow = None
		if keys[K_UP]:
			Y += 10
			follow = None
		if keys[K_DOWN]:
			Y -= 10
			follow = None
		if keys[K_PERIOD]:
			Z += 0.01
			Z = max(0, Z)
			Z = min(10, Z)
		if keys[K_COMMA]:
			Z -= 0.01
			Z = max(0, Z)
			Z = min(10, Z)
		for event in pygame.event.get():
			if event.type == QUIT or event.type == KEYDOWN and event.key == K_ESCAPE:
				if follow is not None:
					follow = None
				else:
					pygame.display.quit()
					pygame.quit()
					display = False
			if event.type == KEYDOWN and event.key == K_SPACE:
				pause = not pause
			if event.type == KEYDOWN and event.key == K_l:
				displayLight = not displayLight
			if event.type == KEYDOWN and event.key == K_s:
				displaySolute = not displaySolute
			if event.type == KEYDOWN and event.key == K_o:
				displayOdor = not displayOdor
			if event.type == KEYDOWN and event.key == K_r:
				seed = True
			if event.type == KEYDOWN and event.key == K_f:
				pygame.display.toggle_fullscreen()
			if event.type == KEYDOWN and event.key == K_x:
				displayCytosol = not displayCytosol
			if event.type == pygame.MOUSEBUTTONUP and event.button == 1: # left=1
				mousePos = pygame.mouse.get_pos()
				mousePos = view_to_world(mousePos)
				point = space.point_query_nearest(mousePos, 0, pymunk.ShapeFilter())
				if point:
					follow = point.shape.Particle
			if event.type == pygame.MOUSEBUTTONUP and event.button == 3: # right=3
				mousePos = pygame.mouse.get_pos()
				mousePos = view_to_world(mousePos)
				particlesToAdd.append(Food(x=mousePos[0], y=mousePos[1]))
