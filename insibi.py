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
bY 		= (-2000, 2000)
solventGrain 	= 50
lightRate 	= 1
odorDegrade 	= 0.98
enzymeDegrade 	= 0.9
poisonDegrade 	= 0.99
nSolvents 	= 7 # odors 1-3, energy/enzyme/poison, light
foodOdor 	= np.array([0,0,0.5])
brown 		= 5
spawnSigma 	= 500
foodInterval 	= 99999999
mutateInterval 	= 100
mutRange 	= 0.05

# particle constants
divisionTime 	= 10
maxAge 		= 999
maxElasticity 	= 0.99
minAttachment 	= 0.01
maxAttachment 	= 0.5
## bonds
maxBonds 	= 6
bondStiff 	= 100
bondDamp 	= 10
bondRatio 	= 20
bondEnRate 	= 10
## genetics
maxActivation 	= 1
nInputs 	= 12 		# OUT: 	energy  light 3solvutes odors1-3  springs_attached N_bondInfo
nInternals 	= 10		# internal layers
nOutputs 	= 13		# IN:   division  elasticicty  N_odor  bonding(break<-0.25..0.75<make)  bondLengthChange  attachment  N_bondInfo  bondEnTrans growChlorophyl
inputLength  	= nInputs + nInternals 
stateLength  	= nInputs + nInternals + nOutputs # values for inputs, internal states and outputs are also aranged like that calculation time
outputLength 	=           nInternals + nOutputs

# cost and rate constants
maxEn 		= 3000
massToEn 	= 100
lightToEn	= 100
rateExist 	= 0.3
enToPoison 	= 0.0001
costDivide 	= 100
rateTransfer 	= 50
costTransfer 	= 1
ratePoison 	= 10
rateChlorophyl	= 1
rateAutotrophy	= 0.00008

# derived constants
wX = bX[1] - bX[0]
wY = bY[1] - bY[0]
solventNx = int( (bX[1] - bX[0]) / solventGrain )
solventNy = int( (bY[1] - bY[0]) / solventGrain )
# sun = (solventNx//2, solventNy//2, 0) # sun is shining at the center of the world

class Particle(object):
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
		if parrent is None: 	# generate random genes and set interaction values to defaut values
			self.weights = 		np.random.rand(outputLength, stateLength) * 2 - 1
			self.biases = 		np.random.rand(outputLength, 1) * 2 - 1 
			self.states = 		np.zeros((stateLength, 1))
			self.outputs = 		np.zeros((outputLength,1))
			self.color = 		np.random.rand(3)
			self.bonding = 		1
			self.bLength = 		1
			self.bStiff = 		1
			self.bDamp = 		1
			self.bInfoRead = 	np.array([0.0,0.0,0.0])
			self.bInfoWrite = 	np.array([0.0,0.0,0.0])
			self.bEnTrans = 	0
			self.chlorophyl = 	100
		else: 			# copy genes and interaction values
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
			self.chlorophyl = 	parrent.chlorophyl
	def step(self):
		# initial checkups
		solvent = get_solvent(self.shape.body.position)
		light = solvent[0]
		light = max(0, min(light, self.chlorophyl * rateAutotrophy))
		solvent[0] -= light
		self.energy += lightToEn * light
		self.energy -= rateExist
		self.energy -= ratePoison * solvent[3]
		self.energy = min(maxEn, self.energy)
		# regulatory network
		## input
		self.states[0,0] 	= self.energy / maxEn 				# IN energy
		self.states[1:8,0] 	= solvent					# IN light, energy, enzyme, poison, odor1-3
		self.states[8,0] 	= len(self.shape.body.constraints) / maxBonds	# IN number bonds
		self.states[9:12,0] 	= self.bInfoRead				# IN information recieved over bonds
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
		growChlorophyl 		= self.outputs[12,0]				# OUT energy sent over bonds
		# adjust body and bond attributes, deposit substances
		self.age += 1
		self._adjust_body()
		self._adjust_bonds()
		solvent[3] += enToPoison * self.energy
		solvent[4:7] += self.odor
		self.bInfoRead.fill(0) # reset bond info buffer
		self.chlorophyl += growChlorophyl
		self.energy -= rateChlorophyl * growChlorophyl
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
		self.color[i] += random.gauss(0, mutRange)
		self.color[i] = min(1, max(0, self.color[i]))
	def check_division(self):
		if (self.energy - costDivide) / 2 < 0: return False
		return self.division > 0.5
	def divide(self):
		self.energy -= costDivide
		self.energy /= 2
		self.chlorophyl /= 2
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
def disperse_solvent(solvent):
	return cv2.blur(solvent, (3,3)) 
def get_solvent(pos):
	x, y = pos
	x = x + wX / 2 # x with 0 at leftmost corner of world
	i = int(x/solventGrain)
	y = y + wY / 2 # y with 0 at leftmost corner of world
	j = int(y/solventGrain)
	return solvent[i,j,:]
def draw_solvent(substance):
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
			if substance == "light":
				light = solvent[i,j,0]
				color = (255*light, 255*light, 255*light)
			if substance == "solute":
				energy = solvent[i,j,1]
				enzyme = solvent[i,j,2]
				poison = solvent[i,j,3]
				color = (255*enzyme, 255*poison, 255*energy)
			if substance == "odor":
				odor1 = solvent[i,j,4]
				odor2 = solvent[i,j,5]
				odor3 = solvent[i,j,6]
				color = (255*odor1, 255*odor2, 255*odor3)
			x = int( (i-i1) * recXwidth )
			y = int( (j-j1) * recYwidth )
			pygame.draw.rect(screen, color, (x,y,recXwidth, recYwidth))

# global data structures
space = pymunk.Space(threaded=True)
space.threads = 2 # max number of threads
space.iterations = 1 # minimum number of iterations for rough but fast collision detection
lines = add_borders(space)
solvent = np.zeros( (solventNx, solventNy, nSolvents) )
solvent[:,:,0] = np.full( (solventNx, solventNy), 0.01)
		
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
		if step % 1000 == 0: # check for living cells regulary
			cells = 0
			for particle in particles:
				if isinstance(particle, Cell): cells += 1
			if cells == 0: # reseed if necessary
				seed = True
			print("\b"*100 + " "*100 + "\b"*100, end="", flush=True)
			print("%8d\tcells:%5d" % (step, cells))

		if seed:
			seed = False
			step = 0
			for i in range(100):
				x = random.gauss(0, spawnSigma)	
				y = random.gauss(0, spawnSigma)	
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

		solvent = disperse_solvent(solvent)
		solvent[solventNx//2-2 : solventNx//2+2 , solventNy//2-2 : solventNy//2+2, 0] 	+= lightRate
		solvent[:,:,2] 	*= enzymeDegrade
		solvent[:,:,3] 	*= poisonDegrade
		solvent[:,:,4:7] *= odorDegrade
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
			follow = None
			X, Y = 0, 0
			Z = 1
		elif inStr == "close" or inStr == "c":
			pygame.display.quit()
			pygame.quit()
			display = False
		elif inStr == "exit" or inStr == "e":
			running = False

# graphic output 
	if display:
		pygame.display.set_caption("GOD\tFPS: %2.2f\tStep: %8d\t[%d:%d] zoom: %2.2f\tCells:%3d" % (clock.get_fps(), step, X, Y, Z, len(particles)))
		screen.fill((0,0,0))
		draw_lines(screen, lines)
		if displayLight:
			draw_solvent("light")
		if displaySolute:
			draw_solvent("solute")
		if displayOdor:
			draw_solvent("odor")
		for particle in particles:
			particle.draw_bonds()
		for particle in particles:
			particle.draw_body()
		if follow:
			X, Y = - follow.shape.body.position
			font = pygame.font.SysFont("monospace", 26)
			if isinstance(follow, Cell):
				string = "E:%4d A:%4d Div:%3.2f Chl:%3.2f Bd:%3.2f BL:%3.2f At:%3.2f El:%3.2f Od:%3.2f:%3.2f:%3.2f" % (follow.energy, follow.age, follow.division, follow.chlorophyl, follow.bonding, follow.bLength, follow.attachment, follow.shape.elasticity, follow.odor[0], follow.odor[1], follow.odor[2])
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
