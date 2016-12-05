#!/home/david/anaconda3/bin/python

import sys, random
import pygame
from pygame.locals import *
import pymunk
import pymunk.pygame_util
from pymunk import Vec2d
import math
import numpy as np
import scipy
from scipy import ndimage
# from numba import jit

bX = (-2000, 2000)
bY = (-2000, 2000)
wX = bX[1] - bX[0]
wY = bY[1] - bY[0]
liquidGrain = 50
liquidNx = int( (bX[1] - bX[0]) / liquidGrain )
liquidNy = int( (bY[1] - bY[0]) / liquidGrain )
dispersion = 1
spawnSigma = 400
foodInterval = 5
mutateInterval = 100
brown = 5
pi = 3.1415

maxElasticity = 0.99
nInternal = 5
states = ["inEnergy", "inSmell", "outDivision", "outElasticity", "outSmell"] + ["internal_%d" % (i+1) for i in range(nInternal)]
nActivations = len(states)
states = {name:i for i, name in enumerate(states)}
mutRange = 0.05
maxAge = 1000
smellDegrade = 0.99

massToEn = 100
maxEn = 3000
rateExist = 0.3
costDivide = 100
rateTransfer = 50
costTransfer = 1


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
	def steal_energy(self, other, energy):
		demand = maxEn - self.energy
		energy = min(energy, demand) # self can't remove more energy than it can swallow
		supply = other.energy
		energy = min(energy, supply) # self can't remove more energy than there is in other
		other.energy -= energy
		self.energy += energy
		self.energy -= costTransfer
	def check_division(self): # dummy function
		return False
	def mutate(self):
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
	def draw(self):
		p = world_to_view(self.shape.body.position)
		pygame.draw.circle(screen, (100,100,100), p, max(2, int(self.shape.radius * Z)), 2)
	def step(self):
		self.age += 1
		self.energy -= rateExist
		self._adjust_attributes()
	def _adjust_attributes(self):
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
		if parrent is None:
			self.weights = 		np.random.rand(nActivations, nActivations) * 2 - 1
			self.biases = 		np.random.rand(nActivations, 1) * 2 - 1
			self.activations = 	np.zeros((nActivations, 1))
			self.color = 		np.random.rand(3)
		else:
			self.shape.body.velocity = parrent.shape.body.velocity
			self.weights = 		np.copy(parrent.weights)
			self.biases = 		np.copy(parrent.biases)
			self.activations = 	np.copy(parrent.activations)
			self.color = 		np.copy(parrent.color)
	def step(self):
		self.age += 1
		# energy handling
		self.energy = min(maxEn, self.energy)
		self.energy -= rateExist
		self._adjust_attributes()
		# regulatory network
		self.activations[states["inEnergy"],0] = self.energy / maxEn
		self.activations[states["inSmell"],0] = get_smell(self.shape.body.position)
		self.activations = np.tanh( np.dot(self.weights, self.activations) + self.biases ) # drop thrugh tanh non-liniearity like in nn
		np.clip(self.activations, -1, 1) # clip to range -1..1
		self.division = self.activations[states["outDivision"],0]
		self.shape.elasticity = min(maxElasticity, self.activations[states["outElasticity"],0] + 1)
		outSmell = self.activations[states["outSmell"],0]
		set_smell(self.shape.body.position, outSmell)
	def mutate(self):
		i = random.randint(0, nActivations-1)
		j = random.randint(0, nActivations-1)
		self.weights[i,j] += random.gauss(0, mutRange)
		i = random.randint(0, nActivations-1)
		self.biases += random.gauss(0, mutRange)
		i = random.randint(0, 2)
		self.color[i] += random.gauss(0, mutRange)
		self.color[i] = min(1, max(0, self.color[i]))
	def check_division(self):
		if (self.energy - costDivide) / 2 < 0: return False
		return self.division > 0
	def divide(self):
# 		print "energy; %d\nweights:\n%s\nbiases:\n%s\n" % (self.energy, self.weights, self.biases)
		self.energy -= costDivide
		self.energy /= 2
		self._adjust_attributes()
		self.age = 0
		x, y = self.shape.body.position
		newCell = Cell(x, y, self.energy, self)
		self.mutate()
		newCell.mutate()
		return newCell
	def draw(self):
		p = world_to_view(self.shape.body.position)
		color = (int(self.color[0]*255), int(self.color[1]*255), int(self.color[2]*255))
		pygame.draw.circle(screen, color, p, max(2, int(self.shape.radius * Z)))
	def _adjust_attributes(self):
		mass = self.energy / massToEn # mass
		mass = max(1, mass)
		self.shape.body.mass = mass
		radius = radius_from_area(self.energy) # radius
		radius = max(1, radius)
		self.shape.unsafe_set_radius(radius)
def radius_from_area(area):
	if area <= 0: return 0
	return math.sqrt(area / pi)
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
	cell.steal_energy(food, rateTransfer)
def disperse_liquid(liquid):
	return scipy.ndimage.filters.gaussian_filter(liquid, dispersion)
def set_smell(pos, smell):
	x, y = pos
	x = x + (bX[1] - bX[0]) / 2 # x with 0 at leftmost corner of world
	i = int(x/liquidGrain)
	y = y + (bY[1] - bY[0]) / 2 # y with 0 at leftmost corner of world
	j = int(y/liquidGrain)
	liquid[i,j] = smell
def get_smell(pos):
	x, y = pos
	x = x + (bX[1] - bX[0]) / 2 # x with 0 at leftmost corner of world
	i = int(x/liquidGrain)
	y = y + (bY[1] - bY[0]) / 2 # y with 0 at leftmost corner of world
	j = int(y/liquidGrain)
	return liquid[i,j]
def draw_liquid():
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
			smell = liquid[i,j]
			color = (255*min(1, max(0, smell)), 255*min(1, max(0, smell)), 255*min(1, max(0, smell)))
			x = int( (i-i1) * recXwidth )
			y = int( (j-j1) * recYwidth )
			pygame.draw.rect(screen, color, (x,y,recXwidth, recYwidth))

pygame.init()
infoObject = pygame.display.Info()
dim = (infoObject.current_w, infoObject.current_h)
screenOffX, screenOffY = infoObject.current_w / 2, infoObject.current_h / 2
X, Y = 0, 0
Z = 1
step = 0

screen = pygame.display.set_mode(dim)
pygame.display.toggle_fullscreen()
clock = pygame.time.Clock()
space = pymunk.Space(threaded=True)
space.threads = 2 # max number of threads
space.iterations = 5
lines = add_borders(space)
liquid = np.zeros( (liquidNx, liquidNy) )
particles = []
colHandler_CF = space.add_collision_handler(1, 2)
colHandler_CF.post_solve = collision_cell_with_food

running = True
draw = True
pause = False
follow = None
displaySmell = True

for i in range(50):
	x = random.gauss(0, spawnSigma)	
	y = random.gauss(0, spawnSigma)	
	particles.append(Cell(x=x, y=y))
for i in range(100):
	x = random.gauss(0, spawnSigma)	
	y = random.gauss(0, spawnSigma)	
	particles.append( Food(x=x, y=y) )
particlesToRemove = []
particlesToAdd = []

while running:
	pygame.display.set_caption("GOD\tFPS: %2.2f\tStep: %8d\t[%d:%d] zoom: %f\t%d" % (clock.get_fps(), step, X, Y, Z, len(particles)))
	if draw:
		screen.fill((0,0,0))
		draw_lines(screen, lines)
		if displaySmell:
			draw_liquid()
		for particle in particles:
			particle.draw()
		if follow:
			X, Y = - follow.shape.body.position
			font = pygame.font.SysFont("monospace", 36)
			string = "Energy: %4d  Age: %4d" % (follow.energy, follow.age)
			text = font.render(string, 1, (255,255,255))
			textpos = text.get_rect()
			textpos.centerx = screen.get_rect().centerx
			screen.blit(text, textpos)
			if follow.check_remove(): follow = None
		pygame.display.flip()

	if not pause:
		if step % foodInterval == 0:
			x = random.gauss(0, spawnSigma)	
			y = random.gauss(0, spawnSigma)	
			particlesToAdd.append(Food(x=x, y=y))
		for particle in particles:
			particle.brownian()
			particle.step()
			if space.current_time_step % mutateInterval == 0:
				particle.mutate()
			if particle.check_division(): 	particlesToAdd.append(particle.divide())
			if particle.check_remove(): 	particlesToRemove.append(particle)
		for particle in particlesToRemove:
			space.remove(particle.shape, particle.shape.body)
			particles.remove(particle)
		particles.extend(particlesToAdd)
		particlesToRemove = []
		particlesToAdd = []

		liquid = disperse_liquid(liquid)
		liquid *= smellDegrade
		np.clip(liquid, 0, 1)
		space.step(1/50.0)
		clock.tick(50)
		step += 1
	
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
		if event.type == QUIT:
			running = False
		if event.type == KEYDOWN and event.key == K_f:
			pygame.display.toggle_fullscreen()
		if event.type == KEYDOWN and event.key == K_SPACE:
			pause = not pause
		if event.type == KEYDOWN and event.key == K_d:
			draw = not draw
		if event.type == KEYDOWN and event.key == K_s:
			displaySmell = not displaySmell
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
		elif event.type == KEYDOWN and event.key == K_ESCAPE:
			running = False
pygame.display.quit()
pygame.quit()
sys.exit(0)
