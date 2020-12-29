import pygame
import random
import math
import neat
import os


pygame.init()
width = 500
height = 700
Gap = 50
Speed = 1
len_gen = 50
wind = pygame.display.set_mode((width, height)) #set the game window
myfont = pygame.font.SysFont('Comic Sans MS', 15)

#-----------------------------------------------------

def distance(x1,y1,x2,y2):
	return math.sqrt((x1-x2)**2 + (y1-y2)**2)

#-----------------------------------------------------

class Pipe:

    def __init__ (self):
        self.x = 0
        self.y = 0
        self.width = random.randint(0,width-(Gap+20))
        self.height = 40


    def draw(self):
        self.y+=Speed
        pygame.draw.rect(wind,(255,0,0),(self.x, self.y,self.width,self.height))
        pygame.draw.rect(wind,(255,0,0),(self.width + Gap, self.y, width - (self.width+Gap), self.height))
        pygame.draw.rect(wind,(0,0,255),(self.x+self.width-10, self.y, 10,10))
        pygame.draw.rect(wind,(0,0,255),(self.width+Gap,self.y, 10,10))
    
    def Gap_space(self):
        return [self.x+self.width-10,self.width+Gap]

#-----------------------------------------------------
       
class Bot:
    def __init__ (self):
        self.x = width/2
        self.y = 670
        self.hit = False
        self.fitness  = 0

    def draw(self):
        if(self.hit == False):
            pygame.draw.rect(wind,(3,252,190),(self.x, self.y,20,20))
        else:
            pygame.draw.rect(wind,(0,0,0),(self.x, self.y,20,20))

    def Right(self):
        if(self.x+1 <= 480):
            self.x+=2
        self.draw()

    def Left(self):
        if(self.x-1 > 0):
            self.x-=2
        self.draw()
    
    def Nothing(self):
        self.x = self.x
        self.draw()

#-----------------------------------------------------

def delete_pipes(array):

    the_dead = []
    for i in range(len(array)):
        if(array[i].y > 700):
            the_dead.append(i)
    the_dead.reverse()
    for j in range(len(the_dead)):
        del array[the_dead[j]]

#-----------------------------------------------------

def closest_pipe(bot,array): # determines closest pipe object to bot 

    distances = []
    pipes = []
    spot = 0

    for i in range(len(array)):
        if(array[i].y < bot.y):
            distances.append(distance(array[i].x,array[i].y,bot.x,bot.y))
            pipes.append(i)

    spot = pipes[distances.index(min(distances))]
   
    return array[spot]

#-----------------------------------------------------

def is_hit(bot,pipe):
    inbetween = False
   
    if(pipe.y + pipe.height>=bot.y>=pipe.y):
        inbetween = True
    elif(pipe.y + pipe.height>=bot.y+20>=pipe.y):
        inbetween = True
    
    if(inbetween):
        points = pipe.Gap_space()
        if(bot.x > points[0] and bot.x+20 < points[1]):
            return False
        else:
            return True
    else:
        return False

#-----------------------------------------------------

def delete_values(Bots,nets,ge):
   
    the_dead = []

    for i in range(len(Bots)):
        if(Bots[i].hit == True):
            the_dead.append(i)

    the_dead.reverse()

    for j in range(len(the_dead)):
        del Bots[the_dead[j]]
        del ge[the_dead[j]]
        del nets[the_dead[j]]
    
#-----------------------------------------------------

def fitness(bot,pipe):

    points = pipe.Gap_space()
    #print("First Gap: ", points[0], " - Second Gap: ", points[1])

    if(bot.x > points[0] and bot.x+20 < points[1]):
        bot.fitness+=10
    if(bot.x < points[0]):
        bot.fitness-=1
    if(bot.x > points[1]):
        bot.fitness-=1
#-----------------------------------------------------

def make_bots(array):
    for i in range(len_gen):
        B = Bot()
        array.append(B)

#-----------------------------------------------------

Fps = 100

def main(genomes,config):
    global Fps
    
    Running = True
    Pipes_ingame = []
    life_pipe = 0

    Bots = []
    nets = []
    ge = []
    
    #creating gene pool
    for _, g in genomes:
        net = neat.nn.FeedForwardNetwork.create(g,config)
        nets.append(net)
        Bots.append(Bot())
        g.fitness = 0
        ge.append(g)

    while Running:

        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    Fps += 100
                if event.key == pygame.K_RIGHT:
                    Fps -= 100
            if event.type == pygame.QUIT:
                Running = False
        

        
        wind.fill((255, 255, 255))
        
        textsurface = myfont.render('Speed: ' + str(Fps),False, (0, 0, 0))
        wind.blit(textsurface,(30,30))

        if(life_pipe%300 == 0): #pipes stay in game for 300 frames
            Pipes_ingame.append(Pipe())

        if(len(Bots) > 0):
            closest = closest_pipe(Bots[0],Pipes_ingame) #determining the closest pipe to a bot

        elif(len(Bots) <= 0): #if there is no bots alive end game run
            Running = False
            break

        for i in range(len(Bots)):

            if(Bots[i].hit == False):
                Bots[i].hit = is_hit(Bots[i],closest)
                fitness(Bots[i],closest)
                ge[i].fitness = Bots[i].fitness
                Gap = closest.Gap_space()
                output = nets[i].activate((Bots[i].x,Gap[0],Gap[1],closest.y)) #input parameters for NEAT

                decision = output.index(max(output)) #

                if(decision == 1):
                    Bots[i].Right()
                elif(decision == 2):
                    Bots[i].Left()
                else:
                    Bots[i].Nothing()

        for i in range(len(Pipes_ingame)):
            Pipes_ingame[i].draw()

        delete_pipes(Pipes_ingame)
        delete_values(Bots,nets,ge)

        pygame.display.update()
        pygame.time.Clock().tick(Fps)
        life_pipe+=1

#-----------------------------------------------------

def run(config_path):
    # displaying statistics and running the NEAT algorithm. 
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)
                        
    p = neat.Population(config)
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    
    winner = p.run(main,100)

   
if __name__ == '__main__':
    #fetching the feedforward NEAT directions
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir,"NEAT_feedforward.txt")
    run(config_path)
