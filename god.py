#!/usr/bin/python

import sys, random
import pygame
from pygame.locals import *
import pymunk
import pymunk.pygame_util
from pymunk import Vec2d
import math
import numpy as np

bX = (-2000, 2000)
bY = (-2000, 2000)
spawnSigma = 400
foodInterval = 10
mutateInterval = 100
brown = 5
pi = 3.1415

maxElasticity = 0.99
nActivations = 8 # IN: energy OUT: division-prob elasticity # 5 internal states
mutRange = 0.05

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
		return exitBorders or noEnergy
	def brownian(self):	
		impulse = (random.randint(-brown,brown), random.randint(-brown,brown) )
		self.shape.body.apply_impulse_at_local_point(impulse)
	def steal_energy(self, energy):
		if self.energy >= energy:
			self.energy -= energy
			return energy
		else:
			self.energy = 0
			return self.energy
	def add_energy(self, energy):
		self.energy += energy
		self.energy -= costTransfer
	def check_division(self): # dummy function
		return False
	def mutate(self):
		pass

class Food(Particle):
	def __init__(self, x=0, y=0, energy=1000):
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
		p = offset(self.shape.body.position)
		pygame.draw.circle(screen, (100,100,100), p, max(2, int(self.shape.radius * Z)), 2)
	def step(self):
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
		# energy handling
		self.energy = min(maxEn, self.energy)
		self.energy -= rateExist
		self._adjust_attributes()
		# regulatory network
		self.activations[0,0] = self.energy / maxEn
		self.activations = np.tanh( np.dot(self.weights, self.activations) + self.biases ) # drop thrugh tanh non-liniearity like in nn
		np.clip(self.activations, -1, 1) # clip to range -1..1
		self.division = self.activations[1,0]
		self.shape.elasticity = min(maxElasticity, self.activations[2,0] + 1)
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
		print "energy; %d\nweights:\n%s\nbiases:\n%s\n" % (self.energy, self.weights, self.biases)
		self.energy -= costDivide
		self.energy /= 2
		self._adjust_attributes()
		x, y = self.shape.body.position
		newCell = Cell(x, y, self.energy, self)
		self.mutate()
		newCell.mutate()
		return newCell
	def draw(self):
		p = offset(self.shape.body.position)
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
def offset(p):
	return int((p.x + X) * Z + screenOffX ), int((p.y + Y) * Z + screenOffY)
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
		p1 = offset(pv1)
		p2 = offset(pv2)
		pygame.draw.lines(screen, (255,255,255), False, [p1,p2])
def collision_cell_with_food(arbiter, space, data):
	a,b = arbiter.shapes
	cell = a.Particle
	food = b.Particle
	energy = food.steal_energy(rateTransfer)
	cell.add_energy(energy)

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
space = pymunk.Space()
lines = add_borders(space)
particles = []
colHandler_CF = space.add_collision_handler(1, 2)
colHandler_CF.post_solve = collision_cell_with_food

running = True
for i in range(10):
	x = random.gauss(0, spawnSigma)	
	y = random.gauss(0, spawnSigma)	
	particles.append(Cell(x=x, y=y))
for i in range(100):
	x = random.gauss(0, spawnSigma)	
	y = random.gauss(0, spawnSigma)	
	particles.append( Food(x=x, y=y) )

while running:
	pygame.display.set_caption("GOD\tFPS: %2.2f\tStep: %8d\t[%d:%d] zoom: %f\t%d" % (clock.get_fps(), step, X, Y, Z, len(particles)))
 	screen.fill((0,0,0))
	draw_lines(screen, lines)

	if step % foodInterval == 0:
		x = random.gauss(0, spawnSigma)	
		y = random.gauss(0, spawnSigma)	
		particles.append(Food(x=x, y=y))

	particlesToRemove = []
	particlesToAdd = []
	for particle in particles:
		particle.brownian()
		particle.step()
		particle.draw()
		if space.current_time_step % mutateInterval == 0:
			particle.mutate()
		if particle.check_division(): 	particlesToAdd.append(particle.divide())
		if particle.check_remove(): 	particlesToRemove.append(particle)
	for particle in particlesToRemove:
		space.remove(particle.shape, particle.shape.body)
		particles.remove(particle)
	particles.extend(particlesToAdd)

	space.step(1/50.0)
	pygame.display.flip()
	clock.tick(50)
	step += 1

	keys = pygame.key.get_pressed()
	if keys[K_RIGHT]:
		X -= 10
	if keys[K_LEFT]:
		X += 10
	if keys[K_UP]:
		Y += 10
	if keys[K_DOWN]:
		Y -= 10
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
		elif event.type == KEYDOWN and event.key == K_ESCAPE:
			running = False
pygame.display.quit()
pygame.quit()
sys.exit(0)
