#!/home/david/anaconda3/bin/python

import pdb
import sys, random
import pymunk
from pymunk import Vec2d
import math
import numpy as np
import scipy
from scipy import ndimage
import cv2
import datetime
import pickle
import readline
import select
# from numba import jit
# random.seed(1)
# np.random.seed(1)

# world constants
bX 		= (-2000, 2000)
bY 		= (-1000, 1000)
solventGrain 	= 50
lightRate 	= 0.005
sunSigmaX	= 7
sunSigmaY	= 7
viscosity 	= 0.1 # viscosity of fluid. Higher values means more viscous - faster objects break stronger
minAttachment 	= 0.01
odorDegrade 	= 0.98
enzymeDegrade 	= 0.9
wasteDegrade 	= 0.99
dispersionRate	= 0.3
nSolvents 	= 7 # odors 1-3, energy/enzyme/waste, light
foodOdor 	= np.array([0,0,0.5])
brown 		= 0
spawnSigma 	= 500 # 200
foodInterval 	= 99999999
mutateInterval 	= 2
mutRange 	= 0.5
colMutRange 	= 0.05 # color change has to be slow
e		= 2.718281828459 # Euler's number

# display constants
shellThickness = 4

# particle constants
maxAge 		= 1000
maxElasticity 	= 0.99
maxChlorophyl	= 1000
maxEnzyme	= 1000
maxTransporter	= 1000
divisionTime	= 100
## bonds
maxBonds 	= 6
bondStiff 	= 100
bondDamp 	= 10
bondRatio 	= 50		# activation to bond length
bondChangeRate	= 1		# change of length per timestep
bondEnRate 	= 10		# bond energy transfer rate per timestep
bondSolRate 	= 0.1		# relative amount of solvent transportable at one timestep
## genetics
maxActivation 	= 1
nInputs 	= 15 		# IN: 	energy  light solutes1-3 odors1-3  springs_attached bondInfo1-3 chlorophyl enzyme transporter
nInternals 	= 20		# internal layers
nOutputs 	= 23		# OUT:   division  elasticicty  N_odor  bonding(break<-0.25..0.75<make)  bondLengthChange  attachment  N_bondInfo  bondEnTrans bondSoluteTrans growChlorophyl growEnzyme growTransporter enzymeExpr enzymeImmob
inputLength  	= nInputs + nInternals 
stateLength  	= nInputs + nInternals + nOutputs # values for inputs, internal states and outputs are also aranged like that calculation time
outputLength 	=           nInternals + nOutputs

# cost and rate constants
maxEn 		= 3000
costExist 	= 1
costWaste 	= 1
costDivide 	= 100
rateTransfer 	= 50
costTransfer 	= 0.00001
## conversions
massToEn 	= 100
lightToEn	= 100
nutrientToEn	= 100
enzymeToEn	= 100
enToWaste 	= 0.0001
## costs for cytosol components
rateChlorophyl	= 0.00001
rateEnzyme	= 0.00001
rateTransporter = 0.00001
## effectiveness of cytosol components
rateHeterotrophy = 0.001
rateAutotrophy	= 0.001
rateExpression 	= 0.001


# derived constants
wX = bX[1] - bX[0]
wY = bY[1] - bY[0]
solventNx = int( (bX[1] - bX[0]) / solventGrain )
solventNy = int( (bY[1] - bY[0]) / solventGrain )
sunX = cv2.getGaussianKernel(solventNx, sunSigmaX)
sunY = cv2.getGaussianKernel(solventNy, sunSigmaY)
sun = sunX.dot(sunY.T)
centerInt = sun[solventNx//2, solventNy//2]
sun = sun / centerInt # make center value 1

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
	def deshape(self):
		self.shape = None
		return None # expand to save shape parameters not allready encoded with energy and states
	def check_division(self): # dummy function
		return False
	def mutate(self):
		pass
	def draw_body(self):
		pass
	def draw_bonds(self):
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
			self.weights = 		np.random.rand(outputLength, stateLength) * 2 - 1
			self.biases = 		np.random.rand(outputLength, 1) * 2 - 1 
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
		## nonlinearity
		self.states[nInputs:,] 	= np.tanh( np.dot(self.weights, self.states) + self.biases ) # nonlinearity
		self.states 		= np.clip(self.states, -maxActivation, maxActivation) 	# clip activations 
		self.outputs 		= self.states[-nOutputs:,]				# make outputs
		self.outputs 		= np.clip(self.outputs, 0, 1) 				# clip outputs
		## output
		self.division 		= self.outputs[0,0] 				# OUT division
		self.odor 		= self.outputs[1:4,0]				# OUT odor
		self.elasticity 	= self.outputs[4,0]		 		# OUT elasticity
		self.bonding 		= self.outputs[5,0]				# OUT bonding propensity
		self.bLength 		= self.outputs[6,0]				# OUT bond length
		self.attachment 	= self.outputs[7,0]				# OUT attachment to surface
		growChlorophyl 		= self.outputs[8,0]				# OUT clorophyl growing
		growEnzyme 		= self.outputs[9,0]				# OUT clorophyl growing
		growTransporter 	= self.outputs[10,0]				# OUT clorophyl growing
		self.bInfoWrite		= self.outputs[11:14,0]				# OUT information sent over bonds
		self.bEnTrans 		= self.outputs[14,0]				# OUT energy sent over bonds
		self.bSolTrans 		= self.outputs[15:21,0]				# OUT solvent and odor transferred over bonds
		self.enzymeExp 		= self.outputs[21,0]				# OUT enzyme expression rate
		self.enzymeImob 	= self.outputs[22,0]				# OUT enzyme surface imobilisation
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
		i = random.randint(0, outputLength-1)
		j = random.randint(0, stateLength-1)
		self.weights[i,j] += random.gauss(0, mutRange)
		# biases
		i = random.randint(0, outputLength-1)
		self.biases[i] += random.gauss(0, mutRange)
		# color
		i = random.randint(0, 2)
		self.color[i] += random.gauss(0, colMutRange)
		self.color[i] = min(1, max(0, self.color[i]))
	def check_division(self):
		if (self.energy - costDivide) / 2 < 0: return False
		return self.division > 0.5
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
		self.mutate()
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
		pygame.draw.circle(screen, color, p, max(shellThickness, int(self.shape.radius * Z)-shellThickness))
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
		if len(self.shape.body.constraints) >= maxBonds or len(other.shape.body.constraints) >= maxBonds: return
		distance = self.shape.body.position.get_distance( other.shape.body.position )
		length = self.shape.radius + other.shape.radius + distance # bond lengths are between surfaces of cells
		spring = pymunk.constraint.DampedSpring(self.shape.body, other.shape.body, (0,0), (0,0), length, bondStiff, bondDamp)
		if self.age < divisionTime: spring.collide_bodies = False # adjecent bodies don't collide -> realistic division behaviour
		space.add(spring)
		spring.cellA = self
		spring.cellB = other
	def _adjust_body(self):
		mass = self.energy / massToEn # mass
		mass = max(1, mass)
		self.shape.body.mass = mass
		radius = radius_from_area(self.energy) # radius
		radius = max(1, radius)
		self.shape.unsafe_set_radius(radius)
		self.shape.elasticity = self.elasticity
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
				if ( self.age > divisionTime and self.bonding < 0.25) or ( other.age > divisionTime and other.bonding < 0.25):
					space.remove(bond)
				if self.age > divisionTime:
					bond.collide_bodies = False # adjecent bodies don't collide -> realistic division behaviour
				targetLength = self.mass*self.bLength*bondRatio + other.mass*other.bLength*bondRatio + self.shape.radius + other.shape.radius # bond lengths are only between surfaces of cells
				bond.rest_length += (targetLength - bond.rest_length) * bondChangeRate 
	def _adjust_forces(self):
		velocVec = self.shape.body.velocity
		if velocVec.length == 0: return
		# fluid resistance of bonds
		if len(self.shape.body.constraints): 
			dragVec = pymunk.vec2d.Vec2d(0, 0)
			for bond in self.shape.body.constraints: # go through all bonds
				posA = bond.cellA.shape.body.position
				posB = bond.cellB.shape.body.position
				bondVec = posA - posB
				if bondVec.length == 0: continue
				bondVec.angle_degrees += 90
				angle = velocVec.get_angle_degrees_between(bondVec)
				if angle > 90 or angle < -90:
					dragVec += bondVec.projection(velocVec)
				else:
					dragVec -= bondVec.projection(velocVec)
			dragVec *= 0.5
			if dragVec.get_length() > velocVec.get_length():
				self.shape.body.velocity.length = 0
			else:
				self.shape.body.velocity.length -= dragVec
		# fluid resistance of body
		viscRatio = float(e ** ( - minAttachment - velocVec.get_length()*viscosity*self.attachment)) 
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
	if cellA.bonding > 0.75 or cellB.bonding > 0.75: cellA.bond(cellB)
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

# global data structures
space = pymunk.Space(threaded=True)
space.threads = 2 # max number of threads
space.iterations = 1 # minimum number of iterations for rough but fast collision detection
lines = add_borders(space)
solvent = np.zeros( (solventNx, solventNy, nSolvents) )
solvent[:,:,0] = np.full( (solventNx, solventNy), 0.2)

# view behaviour
X, Y = 0, 0
Z = 0.45
		
# world behaviour
particles = []
hadler_CF = space.add_collision_handler(1, 2)
hadler_CF.post_solve = collision_cell_with_food
hadler_CC = space.add_collision_handler(1, 1)
hadler_CC.post_solve = collision_cell_with_cell
particlesToRemove = []
particlesToAdd = []
running = True
pause = False
display = False
seed = False
food = False
step = 0

while running:
	if not pause:
		print("step:%8d\tcells:%5d" % (step, len(particles)), end="\r")
		if step % 1000 == 0:
			bonds = [len(list(cell.shape.body.constraints)) for cell in particles]
			if len(particles) > 0:
				print("\t\t\t\tbond-ratio: %2.2f" % (sum(bonds)/float(len(particles))))

		if len(particles) == 0 or seed:
			seed = False
			step = 0
			for i in range(100):
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
			if particle.check_remove(): particlesToRemove.append(particle)
		for particle in particlesToRemove:
			space.remove(particle.shape.body, particle.shape)
			particles.remove(particle)

		particlesToRemove = []
		for particle in particles: 
			particle.brownian()
			particle.step()
			if space.current_time_step % mutateInterval == 0:
				particle.mutate()
			if particle.check_division(): 	particlesToAdd.append(particle.divide())
			if particle.check_remove(): 	particlesToRemove.append(particle)

		for particle in particlesToRemove:
			particles.remove(particle)
			space.remove(particle.shape.body, particle.shape)
		particles.extend(particlesToAdd)
		particlesToAdd = []

		solvent[:,:,0] += 	sun * lightRate
		solvent[:,:,2] *= 	enzymeDegrade
		solvent[:,:,3] *= 	wasteDegrade
		solvent[:,:,4:7] *= 	odorDegrade
		solvent = disperse_solvent(solvent)
		solvent = np.clip(solvent, 0, 1)
		space.step(1/50.0)
		step += 1
		if display:
			clock.tick(50)

# input
	inStr = select.select([sys.stdin,],[],[],0.0)[0] # check if input is pending
	if inStr:
		inStr = input()
		if inStr == "display" or inStr == "d":
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
		elif inStr == "close" or inStr == "c":
			pygame.display.quit()
			pygame.quit()
			display = False
		elif inStr == "exit" or inStr == "e":
			running = False

# graphic output 
	if display:
		pygame.display.set_caption("InSiBi\tFPS: %2.2f\tStep: %8d\t[%d:%d] zoom: %2.2f\tCells: %3d" % (clock.get_fps(), step, X, Y, Z, len(particles)))
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
			font = pygame.font.SysFont("monospace", 20)
			if isinstance(follow, Cell):
				string = "E:%4d A:%4d Div:%3.2f Chl:%3d Enz:%3d EIm:%3.2f EXp: %3.2f Tra:%3d Bd:%3.2f BL:%3.2f At:%3.2f El:%3.2f Od:%3.2f:%3.2f:%3.2f" % (follow.energy, follow.age, follow.division, follow.chlorophyl, follow.enzyme, follow.enzymeImob, follow.enzymeExp, follow.transporter, follow.bonding, follow.bLength, follow.attachment, follow.shape.elasticity, follow.odor[0], follow.odor[1], follow.odor[2])
			elif isinstance(follow, Food):
				string = "E:%4d A:%4d" % (follow.energy, follow.age)
			else: break
			text = font.render(string, 1, (255,255,255))
			textpos = text.get_rect()
			textpos.centerx = screen.get_rect().centerx
			screen.blit(text, textpos)
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
				pygame.display.quit()
				pygame.quit()
				display = False
			if event.type == KEYDOWN and event.key == K_m:
				pygame.display.toggle_fullscreen()
			if event.type == KEYDOWN and event.key == K_SPACE:
				pause = not pause
			if event.type == KEYDOWN and event.key == K_d:
				draw = not draw
			if event.type == KEYDOWN and event.key == K_l:
				displayLight = not displayLight
			if event.type == KEYDOWN and event.key == K_s:
				displaySolute = not displaySolute
			if event.type == KEYDOWN and event.key == K_o:
				displayOdor = not displayOdor
			if event.type == KEYDOWN and event.key == K_r:
				seed = True
			if event.type == KEYDOWN and event.key == K_f:
				fullscreen = not fullscreen
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
