#!/usr/bin/python

import sys, random
import pygame
from pygame.locals import *
import pymunk
import pymunk.pygame_util
from pymunk import Vec2d
import math

bX = (-500, 500)
bY = (-500, 500)
brown = 10
massToEn = 100
pi = 3.1415

rateExist = 0.3
costDivide = 100
rateTransfer = 20
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
	def check_division(self): # dummy function
		return False
	def steal_energy(self, energy):
		if self.energy >= energy:
			self.energy -= energy
			return energy
		else:
			self.erergy = 0
			return self.energy
	def add_energy(self, energy):
		self.energy += energy
		self.energy -= costTransfer

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
		pygame.draw.circle(screen, (255,0,0), p, max(2, int(self.shape.radius * Z)), 2)
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
	def __init__(self, x=0, y=0, energy=10000):
		# body
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
		# soul
		self.divisionProb = 0.003	
	def draw(self):
		p = offset(self.shape.body.position)
		pygame.draw.circle(screen, (0,0,255), p, max(2, int(self.shape.radius * Z)), 2)
	def step(self):
		self.energy -= rateExist
		self._adjust_attributes()
	def check_division(self):
		rand = random.random()
		return rand < self.divisionProb
	def divide(self):
		self.energy -= costDivide
		self.energy /= 2
		self._adjust_attributes()
		x, y = self.shape.body.position
		newCell = Cell(x, y, self.energy)
		return newCell
	def _adjust_attributes(self):
		mass = self.energy / massToEn
		radius = radius_from_area(self.energy)
		mass = max(1, mass)
		radius = max(1, radius)
		self.shape.unsafe_set_radius(radius)
		self.shape.body.mass = mass
		

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
Z = 0.85

screen = pygame.display.set_mode(dim)
pygame.display.toggle_fullscreen()
clock = pygame.time.Clock()
space = pymunk.Space()
lines = add_borders(space)
particles = []
colHandler_CF = space.add_collision_handler(1, 2)
colHandler_CF.post_solve = collision_cell_with_food

running = True
particles.append(Cell())
ticks_to_next_food = 25

while running:
	pygame.display.set_caption("GOD\tFPS: %2.2f\t[%d:%d] zoom: %f\t%d" % (clock.get_fps(), X, Y, Z, len(particles)))
 	screen.fill((0,0,0))
	draw_lines(screen, lines)

	ticks_to_next_food -= 1
	if ticks_to_next_food <= 0:
		ticks_to_next_food = 20
		particle = Food(x = random.randint(bX[0],bX[1]), y = random.randint(bY[0], bY[1]))
		particles.append(particle)

	particlesToRemove = []
	particlesToAdd = []
	for particle in particles:
		particle.brownian()
		particle.step()
		particle.draw()
		if particle.check_remove(): 	particlesToRemove.append(particle)
		if particle.check_division(): 	particlesToAdd.append(particle.divide())
	for particle in particlesToRemove:
		space.remove(particle.shape, particle.shape.body)
		particles.remove(particle)
	particles.extend(particlesToAdd)
	
	space.step(1/50.0)
	pygame.display.flip()
	clock.tick(50)

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
