import sys, random
import pygame
from pygame.locals import *
import pymunk
import pymunk.pygame_util
from pymunk import Vec2d

bX = (-500, 500)
bY = (-500, 500)

def offset(p):
	return int((p.x + X) * Z + screenOffX ), int((p.y + Y) * Z + screenOffY)

def add_ball(space):
	"""Add a ball to the given space at a random position"""
	mass = 1
	radius = 14
	inertia = pymunk.moment_for_circle(mass, 0, radius, (0,0))
	body = pymunk.Body(mass, inertia)
	x = random.randint(-200,200)
	y = random.randint(-200,200)
	body.position = x, y
	shape = pymunk.Circle(body, radius, (0,0))
	space.add(body, shape)
	return shape

def add_borders(space):
	body = pymunk.Body(body_type = pymunk.Body.STATIC)
	body.position = (0, 0)
	l1 = pymunk.Segment(body, (bX[0], bY[0]),  (bX[0], bY[1]), 5)
	l2 = pymunk.Segment(body, (bX[0], bY[0]),  (bX[1], bY[0]), 5)
	l3 = pymunk.Segment(body, (bX[1], bY[1]),  (bX[0], bY[1]), 5)
	l4 = pymunk.Segment(body, (bX[1], bY[1]),  (bX[1], bY[0]), 5)
	space.add(l1, l2, l3, l4)
	return l1,l2, l3, l4

def draw_ball(screen, ball):
	p = offset(ball.body.position)
	pygame.draw.circle(screen, (0,0,255), p, max(2, int(ball.radius * Z)), 2)

def draw_lines(screen, lines):
	for line in lines:
		body = line.body
		pv1 = body.position + line.a.rotated(body.angle)
		pv2 = body.position + line.b.rotated(body.angle)
		p1 = offset(pv1)
		p2 = offset(pv2)
		pygame.draw.lines(screen, (255,255,255), False, [p1,p2])


pygame.init()
infoObject = pygame.display.Info()
dim = (infoObject.current_w, infoObject.current_h)
screenOffX, screenOffY = infoObject.current_w / 2, infoObject.current_h / 2
X, Y = 0, 0
Z = 1
screen = pygame.display.set_mode(dim)
clock = pygame.time.Clock()
space = pymunk.Space()
myfont = pygame.font.SysFont("monospace", 15)

lines = add_borders(space)
balls = []
draw_options = pymunk.pygame_util.DrawOptions(screen)

ticks_to_next_ball = 0.1
while True:
	pygame.display.set_caption("GOD [%d:%d] zoom: %f\t%d" % (X, Y, Z, len(balls)))


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
			sys.exit(0)
		if event.type == KEYDOWN and event.key == K_f:
			pygame.display.toggle_fullscreen()
		elif event.type == KEYDOWN and event.key == K_ESCAPE:
			sys.exit(0)

	ticks_to_next_ball -= 1
	if ticks_to_next_ball <= 0:
		ticks_to_next_ball = 25
		ball_shape = add_ball(space)
		balls.append(ball_shape)

 	screen.fill((0,0,0))
	
	draw_lines(screen, lines)

	balls_to_remove = []
	for ball in balls:
		impulse = (random.randint(-5,5), random.randint(-5,5) )
		ball.body.apply_impulse_at_local_point(impulse)
		if ball.body.position.x < bX[0] or ball.body.position.y < bY[0] or ball.body.position.x > bX[1] or ball.body.position.y > bY[1]:
			balls_to_remove.append(ball)
		draw_ball(screen, ball)

	for ball in balls_to_remove:
		space.remove(ball, ball.body)
		balls.remove(ball)
	
	space.step(1/50.0)

	pygame.display.flip()
	clock.tick(50)
