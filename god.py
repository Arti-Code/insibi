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
random.seed(1)
np.random.seed(1)

# world constants
bX = (-2000, 2000)
bY = (-2000, 2000)
liquidGrain = 50
nOdors = 3
odorDegrade = 0.98
nLiquids = nOdors + 2
foodOdor = np.array([0,0,0.5])
dispersion = 1
brown = 5
spawnSigma = 500
foodInterval = 3
mutateInterval = 100
mutRange = 0.05

# particle constants
nInfo = 3
nInputs = 1 + nOdors + 1 + nInfo	# OUT: 	energy  N_odor  springs_attached N_bondInfo
nInternals = 10				# internal layers
nOutputs = 2 + nOdors + 3 + nInfo + 1	# IN:   division  elasticicty  N_odor  bonding(break<-0.25..0.75<make)  bondLengthChange  attachment  N_bondInfo  bondEnTrans
inputLength  = nInputs + nInternals 
stateLength  = nInputs + nInternals + nOutputs # values for inputs, internal states and outputs are also aranged like that calculation time
outputLength =           nInternals + nOutputs
maxAge = 999
maxActivation = 1
maxElasticity = 0.99
maxBonds = 6
bondRatio = 20
bondStiff = 100
bondDamp = 10
bondEnRate = 10
divisionTime = 10
minAttachment = 0.01
maxAttachment = 0.5

# cost and rate constants
massToEn = 100
maxEn = 3000
rateExist = 0.3
costDivide = 100
rateTransfer = 50
costTransfer = 1

# for save function
constants = {"bX":bX, "bY":bY, "liquidGrain":liquidGrain, "dispersion":dispersion, "odorDegrade": odorDegrade, "brown": brown,
	"spawnSigma":spawnSigma, "foodInterval":foodInterval, "maxElasticity":maxElasticity, "stateLength":stateLength, 
	"mutateInterval": mutateInterval, "mutRange":mutRange, "maxAge":maxAge, "massToEn":massToEn, "rateExist":rateExist,
	"costDivide":costDivide, "rateTransfer":rateTransfer, "costTransfer":costTransfer}

# derived constants
wX = bX[1] - bX[0]
wY = bY[1] - bY[0]
liquidNx = int( (bX[1] - bX[0]) / liquidGrain )
liquidNy = int( (bY[1] - bY[0]) / liquidGrain )

class Particle(object):
# 	def __del__(self):
# 		for bond in self.bonds:
# 			for cell in [bond.cellA, bond.cellB]:
# 				if cell != self:
# 					cell.remove_bond(bond)
# 			self.remove_bond(bond)
# 			space.remove(bond)
# 		space.remove(self.shape, self.shape.body)
	def check_remove(self):
		x = self.shape.body.position.x
		y = self.shape.body.position.y
		exitBorders = x < bX[0] or y < bY[0] or x > bX[1] or y > bY[1]
		noEnergy = self.energy <= 0
		oldAge = self.age > maxAge
		return exitBorders or noEnergy or oldAge
	def brownian(self):	
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
	def remove_bond(self, delBond):
		removeBond = None
		for bond in self.bonds:
			if bond == delBond: removeBond = delBond
		self.bonds.remove(delBond)
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
		self.bonds = []
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
		self.energy -= rateExist
		self._adjust_body()
		set_odor(self.shape.body.position, foodOdor) # OUT odor
	def _adjust_body(self):
		mass = self.energy / massToEn
		radius = radius_from_area(self.energy)
		mass = max(1, mass)
		radius = max(1, radius)
		self.shape.unsafe_set_radius(radius)
		self.shape.body.mass = mass

class Cell(Particle):
	def __init__(self, x=0, y=0, energy=1000, parrent=None):
		self.age = 0
		self.energy = energy
		mass = max(1, self.energy / massToEn)
		radius = max(1, radius_from_area(self.energy))
		inertia = pymunk.moment_for_circle(mass, 0, radius, (0,0))
		body = pymunk.Body(mass, inertia)
		body.position = x, y
		shape = pymunk.Circle(body, radius, (0,0))
		space.add(body, shape)
		self.shape = shape
		self.shape.Particle = self
		self.shape.collision_type = 1
		if parrent is None: # generate random genes and set interaction values to defaut values
			self.weights = 		np.random.rand(outputLength, stateLength) * 2 - 1
			self.biases = 		np.random.rand(outputLength, 1) * 2 - 1 
			self.states = 		np.zeros((stateLength, 1))
			self.outputs = 		np.zeros((outputLength,1))
			self.color = 		np.random.rand(3)
			self.bonding = 1
			self.bLength = 1
			self.bStiff = 1
			self.bDamp = 1
			self.bInfoRead = np.array([0.0,0.0,0.0])
			self.bInfoWrite = np.array([0.0,0.0,0.0])
			self.bEnTrans = 0
		else: # copy genes and interaction values
			self.shape.body.velocity = parrent.shape.body.velocity
			self.weights = 		np.copy(parrent.weights)
			self.biases = 		np.copy(parrent.biases)
			self.states = 		np.copy(parrent.states)
			self.outputs = 		np.copy(parrent.outputs)
			self.color = 		np.copy(parrent.color)
			self.bonding = 		parrent.bonding
			self.bLength = 		parrent.bLength
			self.bStiff = 		parrent.bStiff
			self.bDamp = 		parrent.bDamp
			self.bInfoRead =	parrent.bInfoRead 
			self.bInfoWrite = 	parrent.bInfoWrite
			self.bEnTrans = 	parrent.bEnTrans
	def step(self):
		# initial checkups
		self.energy = min(maxEn, self.energy)
		self.energy -= rateExist
		# regulatory network
		## input
		self.states[0,0] 	= self.energy / maxEn 				# IN energy
		self.states[1:4,0] 	= get_odor(self.shape.body.position)		# IN odor
		self.states[4,0] 	= len(self.shape.body.constraints) / maxBonds	# IN number bonds
		self.states[5:8,0] 	= self.bInfoRead				# IN information recieved over bonds
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
		self.bInfoWrite		= self.outputs[8:11,0]				# OUT information sent over bonds
		self.bEnTrans 		= self.outputs[11,0]				# OUT energy sent over bonds
		# adjust body and bond attributes, deposit odor
		self.age += 1
		self._adjust_body()
		self._adjust_bonds()
		set_odor(self.shape.body.position, self.odor)
		self.bInfoRead.fill(0)
	def mutate(self):
		i = random.randint(0, outputLength-1)
		j = random.randint(0, inputLength-1)
		self.weights[i,j] += random.gauss(0, mutRange)
		i = random.randint(0, outputLength-1)
		self.biases += random.gauss(0, mutRange)
		i = random.randint(0, 2)
		self.color[i] += random.gauss(0, mutRange)
		self.color[i] = min(1, max(0, self.color[i]))
	def check_division(self):
		if (self.energy - costDivide) / 2 < 0: return False
		return self.division > 0.5
	def divide(self):
		self.energy -= costDivide
		self.energy /= 2
		self._adjust_body()
		self.age = 0
		x, y = self.shape.body.position
		newCell = Cell(x, y, self.energy, self)
		self.mutate()
		newCell.mutate()
		self.bond(newCell)
		return newCell
	def draw_body(self):
		p = world_to_view(self.shape.body.position)
		color = (int(self.color[0]*255), int(self.color[1]*255), int(self.color[2]*255))
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
		if len(self.shape.body.constraints) >= maxBonds or len(other.shape.body.constraints) >= maxBonds: return
		length = self.bLength*bondRatio + other.bLength*bondRatio + self.shape.radius + other.shape.radius # bond lengths are only between surfaces
		spring = pymunk.constraint.DampedSpring(self.shape.body, other.shape.body, (0,0), (0,0), length, bondStiff, bondDamp)
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
		self.shape.body.velocity = self.shape.body.velocity * (1-float(self.attachment)*maxAttachment - minAttachment)
		self.shape.elasticity = self.elasticity
	def _adjust_bonds(self):
		bondsToRemove = []
		for bond in self.shape.body.constraints:
			cellA = bond.cellA
			cellB = bond.cellB
			if not isinstance(cellA, Cell)or not isinstance(cellB, Cell):
				space.remove(bond)
			if cellA == self: other = cellB
			else: other = cellA
			if self.age > divisionTime and (other.energy < 0 or self.bonding < 0.25):
				space.remove(bond)
			bond.rest_length = self.bLength*bondRatio + other.bLength*bondRatio + self.shape.radius + other.shape.radius # bond lengths are only between surfaces
			other.bInfoRead += self.bInfoWrite
			self.bInfoRead += other.bInfoRead
			self.transfer_energy(other, self.bEnTrans * bondEnRate)
			
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
	if cellA.bonding < 0.75 or cellB.bonding < 0.75: return
	cellA.bond(cellB)
def disperse_liquid(liquid):
	return cv2.blur(liquid, (3,3)) 
def set_odor(pos, odor):
	x, y = pos
	x = x + (bX[1] - bX[0]) / 2 # x with 0 at leftmost corner of world
	i = int(x/liquidGrain)
	y = y + (bY[1] - bY[0]) / 2 # y with 0 at leftmost corner of world
	j = int(y/liquidGrain)
	liquid[i,j,:3] = odor
def get_odor(pos):
	x, y = pos
	x = x + wX / 2 # x with 0 at leftmost corner of world
	i = int(x/liquidGrain)
	y = y + wY / 2 # y with 0 at leftmost corner of world
	j = int(y/liquidGrain)
	return liquid[i,j,:3]
def draw_odor():
	x1, y1 = view_to_world( (0, 0) ) # get points for rectangle of world coordinates displayed on screen
	x2, y2 = view_to_world( (screenOffX*2,screenOffY*2) )
	x1 = x1 + wX/2 # de-center world coordinates, so that top-right of map is 0,0
	y1 = y1 + wY/2
	x2 = x2 + wX/2
	y2 = y2 + wY/2
	i1 = max(0, min(liquidNx, int(x1/liquidGrain) )) # get indices for liquid
	j1 = max(0, min(liquidNy, int(y1/liquidGrain) ))
	i2 = max(0, min(liquidNx, int(x2/liquidGrain) ))
	j2 = max(0, min(liquidNy, int(y2/liquidGrain) ))
	recXwidth = int( (screenOffX*2) / (i2-i1) )
	recYwidth = int( (screenOffY*2) / (j2-j1) )
	for i in range(i1,i2):
		for j in range(j1, j2):
			odors = []
			for k in range(nOdors):
				odors.append(liquid[i,j,k])
			color = (255*min(1, max(0, odors[0])), 255*min(1, max(0, odors[1])), 255*min(1, max(0, odors[2])))
			x = int( (i-i1) * recXwidth )
			y = int( (j-j1) * recYwidth )
			pygame.draw.rect(screen, color, (x,y,recXwidth, recYwidth))
def save(constants, step, liquid, particles, saveFile):
	with open(saveFile, 'wb') as file:
		pickle.dump(constants, file, protocol=pickle.HIGHEST_PROTOCOL)
		pickle.dump(step, file, protocol=pickle.HIGHEST_PROTOCOL)
		pickle.dump(liquid, file, protocol=pickle.HIGHEST_PROTOCOL)
		for particle in particles:
			tmpShape = particle.shape
			shapeVals = particle.deshape()
			pickle.dump(particle, file, protocol=pickle.HIGHEST_PROTOCOL)
			particle.shape = tmpShape
		
space = pymunk.Space(threaded=True)
space.threads = 2 # max number of threads
space.iterations = 1 # minimum number of iterations for rough but fast collision detection
lines = add_borders(space)
liquid = np.zeros( (liquidNx, liquidNy, nLiquids) )
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
step = 0

while running:
	if not pause:
# 		if step % 100000 == 0 and step != 0: # save world
# 			saveFile = "saves/%s_%s.god" % (datetime.date.today(), step)
# 			save(constants, step, liquid, particles, saveFile)
		if step % 1000 == 0: # check for living cells regulary
			food, cells = 0, 0
			for particle in particles:
				if isinstance(particle, Cell): cells += 1
				if isinstance(particle, Food): food += 1
			if cells == 0: # reseed if necessary
				seed = True
			print("\b"*100 + " "*100 + "\b"*100, end="", flush=True)
			print("%8d\tcells:%5d\tfood:%5d" % (step, cells, food))
		if step % foodInterval == 0: # spawn new food
			x = random.gauss(0, spawnSigma)	
			y = random.gauss(0, spawnSigma)	
			particlesToAdd.append(Food(x=x, y=y))
		if seed:
			seed = False
			step = 0
			for i in range(100):
				x = random.gauss(0, spawnSigma)	
				y = random.gauss(0, spawnSigma)	
				particles.append(Cell(x=x, y=y))
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

		liquid = disperse_liquid(liquid)
		liquid *= odorDegrade
		np.clip(liquid, 0, 1)
		space.step(1/50.0)
		step += 1
		if display:
			clock.tick(50)

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
# 			pygame.display.toggle_fullscreen()
			clock = pygame.time.Clock()
			follow = None
			displayOdor = False
			display = True
			X, Y = 0, 0
			Z = 1
		elif inStr == "close" or inStr == "c":
			pygame.display.quit()
			pygame.quit()
			display = False
		elif inStr == "exit" or inStr == "e":
			running = False
	if display:
		pygame.display.set_caption("GOD\tFPS: %2.2f\tStep: %8d\t[%d:%d] zoom: %2.2f\t%d" % (clock.get_fps(), step, X, Y, Z, len(particles)))
		screen.fill((0,0,0))
		draw_lines(screen, lines)
		if displayOdor:
			draw_odor()
		for particle in particles:
			particle.draw_bonds()
		for particle in particles:
			particle.draw_body()
		if follow:
			X, Y = - follow.shape.body.position
			font = pygame.font.SysFont("monospace", 26)
			if isinstance(follow, Cell):
				string = "E:%4d A:%4d Div:%3.2f Bd:%3.2f BL:%3.2f At:%3.2f El:%3.2f Od:%3.2f:%3.2f:%3.2f" % (follow.energy, follow.age, follow.division, follow.bonding, follow.bLength, follow.attachment, follow.shape.elasticity, follow.odor[0], follow.odor[1], follow.odor[2])
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
			if event.type == KEYDOWN and event.key == K_f:
				pygame.display.toggle_fullscreen()
			if event.type == KEYDOWN and event.key == K_SPACE:
				pause = not pause
			if event.type == KEYDOWN and event.key == K_d:
				draw = not draw
			if event.type == KEYDOWN and event.key == K_o:
				displayOdor = not displayOdor
			if event.type == KEYDOWN and event.key == K_s:
				seed = True
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
