from typing import Dict
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.artist import Artist
import struct
import sys

def struct_isqrt(number):
    threehalfs = 1.5
    x2 = number * 0.5
    y = number
    
    packed_y = struct.pack('f', y)       
    i = struct.unpack('i', packed_y)[0]  # treat float's bytes as int 
    i = 0x5f3759df - (i >> 1)            # arithmetic with magic number
    packed_i = struct.pack('i', i)
    y = struct.unpack('f', packed_i)[0]  # treat int's bytes as float
    
    y = y * (threehalfs - (x2 * y * y))  # Newton's method
    return y



## PODE MUDAR, VAI TESTANDO AS DIFERENTES CONSTANTES

DIMENSION = 50 ## MAIS DIMENSAO, MAIS DETALHADO MAS TB MAIS DEVAGAR
INTENSITY_CLICK = 1000000 ## A INTENSIDADE DE CADA CLICK
LIMIT_VIEW_INTENSITY = 10000 ## A PARTIR DESSA INTENSIDADE, A LUMINOSIDADE NA TELA É 100%, OU SEJA, SE A INTENSIDADE É 50%
BLOCK_VISABILITY = LIMIT_VIEW_INTENSITY/2
fps = 32

## MAPA INICIAL: FACA SEU MAPA FAZENDO BLOCOS, (tamanho_y, tamanho_x, pos_top_esquerda_y (1 até DIMENSION-1), pos_top_esquerda_x(1 até DIMENSION-1))
blockList=[(15,20, 5, 10), (10,20,23,20)]


textPos = (0,DIMENSION+1)
textDimPos = (30,DIMENSION+1)
intTextPos = (10,DIMENSION+1)
intTutorialPos = (0, -4)

###############################################################################################
### N MEXE NAS CONSTANTES DAQUI EM BAIXO
###############################################################################################

## CONSTANTES
angleInicial = 22.5 * np.pi / 180 
cteRaiz = np.sin(angleInicial)
ctePi = np.pi/4
raiz2 = np.sqrt(2)/2
intensidadeMin = 0.1
listRotate = [[1,1],[1,0],[1,-1],[0,-1],[-1,-1],[-1,0],[-1,1],[0,1]]

# Direction (x,y)
#   ____ y
#  |
#  |x

# Unica função q pode melhorar
def addIntensity(X,Y, grid, direction, I):
    return addTwoIntensities(grid[X + direction[0],Y + direction[1]], createVector(I, direction)) # Creating the vector with direction is an imprecision



def addTwoIntensities(i1, i2):
    I_a = int(i1['intensity'])
    k_a = normalizeDirection(i1['direction'])
    I_d = int(i2['intensity'])
    k_d = normalizeDirection(i2['direction'])
        
    newVector = (I_a*k_a[0] + I_d*k_d[0], I_a*k_a[1] + I_d*k_d[1])
    if (newVector[0] == 0 and newVector[1] == 0):
        return createVector()
    invNewIntensity = struct_isqrt(newVector[0]**2 + newVector[1]**2)
    newDirection = (newVector[0]*invNewIntensity, newVector[1]*invNewIntensity)
    return createVector(1/invNewIntensity, newDirection)

def normalizeDirection(direction):
    dx = direction[0]
    dy = direction[1]
    normalize = struct_isqrt(dx**2 + dy**2)
    if (dx!= 0 or dy!= 0) :
        dx = dx*normalize
        dy = dy*normalize
    return [dx, dy]

def mainDirection(direction):
    direction = normalizeDirection(direction)
    xsum=0
    ysum=0
    if direction[1] > cteRaiz:
        ysum += 1
    if direction[1] < -cteRaiz:
        ysum -= 1
    if direction[0] > cteRaiz:
        xsum += 1
    if direction[0] < -cteRaiz:
        xsum -= 1
    return [xsum,ysum]

def calcIntensity(a1,a2):
    return np.abs((np.sin(a1) + np.sin(a2))/2)

def mainIntensity(angle, intensidade):
    return intensidade*calcIntensity(angle, ctePi - angle)

def intensidadeInferior(angle, intensidade):
    return intensidade*calcIntensity(-angle, angle + ctePi)

def intensidadeSuperior(angle, intensidade):
    return intensidade*calcIntensity(ctePi - angle, angle - 2*ctePi)

def intensidadeInfInferior(angle, intensidade):
    return intensidade*calcIntensity(-(angle+ctePi), ctePi*2)

def intensidadeSupSuperior(angle, intensidade):
    return intensidade*calcIntensity(angle-2*ctePi, ctePi*2)

def obterAnguloRelativo(direction):
    direction = normalizeDirection(direction)
    angle = np.arcsin(direction[1])
    if direction[0] < 0:
        angle = (np.pi - angle) % (2 * np.pi)
    return (angle - angleInicial) % (np.pi/4)
    
def createVector(intensity=0, direction=[0,0], isBlock=False):
    return {'intensity': intensity, 'direction': direction, 'isBlock': isBlock}

def modifyBlocks(grid, X, Y, newGrid):
    intensidade = grid[X,Y]['intensity']
    direction = grid[X,Y]['direction']
    
    if grid[X,Y]['isBlock'] == True or intensidade <= intensidadeMin:
        return
        
    if direction == [0,0]:
        newint = intensidade/8
        sumNonDirectional(newint, grid, newGrid, X, Y)
    else:
        angle = obterAnguloRelativo(direction)
        mainInt = mainIntensity(angle, intensidade)
        intInf = intensidadeInferior(angle, intensidade)
        intintInf = intensidadeInfInferior(angle, intensidade)
        intSup = intensidadeSuperior(angle, intensidade)
        intsupSup = intensidadeSupSuperior(angle, intensidade)
        mainDirect = mainDirection(direction)
        sumDirectional(mainInt, intInf, intintInf, intSup, intsupSup, X, Y, mainDirect, grid, newGrid)

def sumDirectional(intMain, intInf,intintInf, intSup, intsupSup, X, Y, mainDirect, grid, newGrid):
    indexMain = listRotate.index(mainDirect)
    listInt = [intsupSup, intSup, intMain, intInf, intintInf]
    for i in range(-2, 3):
        index = indexMain+i
        if index > len(listRotate) - 1:
            index -= len(listRotate)
        direction = listRotate[index].copy()
        if grid[X + direction[0],Y]['isBlock'] == True:
            direction[0] = -direction[0]
        if grid[X,Y + direction[1]]['isBlock'] == True:
            direction[1] = -direction[1]
        intensity = listInt[i+2]
        newGrid[X + direction[0], Y + direction[1]] = addIntensity(X ,Y,newGrid,direction,intensity)
          
def sumNonDirectional(newint, grid, newGrid, X, Y):
    for direction in listRotate:
        direction = direction.copy()
        if grid[X + direction[0],Y]['isBlock'] == True:
            direction[0] = -direction[0]
        if grid[X,Y + direction[1]]['isBlock'] == True:
            direction[1] = -direction[1]
        newGrid[X + direction[0], Y + direction[1]] = addIntensity(X,Y,grid,direction,newint)

def makeHollowRectangle(grid, a, b, x, y):
    makeRectangle(grid, a, 1, x,y)
    makeRectangle(grid, a, 1, x, y + b - 1)
    makeRectangle(grid, 1, b, x,y)
    makeRectangle(grid, 1, b, x + a - 1,y)

def makeRectangle(grid, a ,b, x, y):
    for i_x in range(a):
        for i_y in range(b):
            try:
                grid[x +i_x, y+i_y] = createVector(isBlock=True)
            except:
                pass
def initializeGrid():
    grid = np.full((DIMENSION, DIMENSION), {})
    for x in range(DIMENSION):
        for y in range(DIMENSION):
            grid[x,y] = createVector()
    return placeBlocks(grid)
        
def mainLoop(grid):
    newGrid = initializeGrid()
    for x in range(DIMENSION):
        for y in range(DIMENSION):
            modifyBlocks(grid,x,y,newGrid)
    placeBlocks(newGrid)
    return newGrid

def placeBlocks(grid):
    makeHollowRectangle(grid, DIMENSION, DIMENSION, 0,0)
    for block in blockList:
        makeRectangle(grid,*block)
    
    return grid

def translateGrid(grid, block=0):
    newGrid = np.zeros((DIMENSION,DIMENSION), dtype=np.int64)
    try:
        for x in range(DIMENSION):
            for y in range(DIMENSION):
                if grid[x,y]['isBlock']:
                    newGrid[x,y] = block
                else:
                    intensity = int(grid[x,y]['intensity'])
                    
                    if intensity > LIMIT_VIEW_INTENSITY:
                        intensity = LIMIT_VIEW_INTENSITY
                    newGrid[x,y] = intensity
    except:
        print('error')
        print(grid)
    return newGrid

indexprint = 0

def translateDirection(grid):
    newGrid = np.full((DIMENSION,DIMENSION), {})
    for x in range(DIMENSION):
        for y in range(DIMENSION):
            if grid[x,y]['isBlock'] == False:
                newGrid[x,y] = (round(grid[x,y]['direction'][0], 2),round(grid[x,y]['direction'][1], 2))
            else:
                newGrid[x,y] = (0,0)
    return newGrid

def printGrid(grid):
    global indexprint
    trans = translateGrid(grid)
    # print(f'{indexprint}: \n',trans)
    plt.imshow(trans, cmap='gray')
    plt.show()
    indexprint +=1

pressMode = 0
firstClick = (0,0)
secondClick = (0,0)

def on_pick(event):
    global newGrid, pressMode, firstClick, secondClick
    try:
        if event.button == 1:
            x = round(event.xdata)
            y = round(event.ydata)
            if x == DIMENSION or x==0 or y== DIMENSION or y==0:
                return
            for i in range(-1,2,1):
                for j in range(-1,2,1):
                    if not newGrid[y+i,x+j]['isBlock']:
                        newGrid[y+i,x+j]['intensity'] += INTENSITY_CLICK
        if event.button == 3:
            x = round(event.xdata)
            y = round(event.ydata)
            if pressMode == 0:
                pressMode = 1
                firstClick = (x,y)
            else:
                pressMode = 0
                secondClick = (x,y)
                generateBlock()
        if event.button == 2:
            resetBlocks()
    except:
        pass

def on_press(event):
    global fps, text, DIMENSION, INTENSITY_CLICK, intensityText
    sys.stdout.flush()
    if event.key == 'w':
        fps += 2
        text = removeFPS(text)

    elif event.key == 'e':
        fps -= 2
        text = removeFPS(text)
    
    elif event.key == 'y':
        INTENSITY_CLICK += 10000
        intensityText = removeIntText(intensityText)
    elif event.key == 'u':
        if INTENSITY_CLICK > LIMIT_VIEW_INTENSITY + 10000:
            INTENSITY_CLICK -= 10000
        intensityText = removeIntText(intensityText)


def writeFPS():
    global fps
    return plt.text(*textPos,f'FPS IDEAL: {fps}', size=10, ha="left", va="center")
def removeFPS(text):
    Artist.remove(text)
    return writeFPS()

def writeDIMENSION():
    global DIMENSION
    return plt.text(*textDimPos,f'Dimensão do Grid: {DIMENSION}', size=10, ha="left", va="center")

def writeINTENSITY():
    global INTENSITY_CLICK, intTextPos
    return plt.text(*intTextPos,f'Intensidade de Click: {INTENSITY_CLICK}', size=10, ha="left", va="center")
def removeIntText(text):
    Artist.remove(text)
    return writeINTENSITY()

def writeTutorial():
    return plt.text(*intTutorialPos, 'Use "w" para aumentar FPS e "e" para diminuir FPS, use "y" para aumentar intensidade do click\ne "u" para reduzir ela. Botão esquerdo do mouse faz uma onda, botão do meio remove os\nobstaculos e botão direito clica duas vezes para fazer obstaculos. Para mudar dimensão, \nsó mudar manualmente no inicio do código num editor de texto', size=10, ha="left", va="center", color='green')

def resetBlocks():
    global blockList
    blockList = []

def generateBlock():
    x_1 = firstClick[0]
    x_2 = secondClick[0]
    listX = (x_1, x_2) if x_1 < x_2 else (x_2,x_1)
    tamanhoX = listX[1] - listX[0] + 1
    y_1 = firstClick[1]
    y_2 = secondClick[1]
    listY = (y_1, y_2) if y_1 < y_2 else (y_2,y_1)
    tamanhoY = listY[1] - listY[0] + 1
    blockList.append((tamanhoY, tamanhoX, listY[0], listX[0]))


newGrid = initializeGrid()
newGrid[1,1]['intensity'] = 1000000 ## Inicial pulse

nSeconds = 10
a = translateGrid(newGrid)
fig = plt.figure( figsize=(8,8) )
im = plt.imshow(a,cmap='gray')
fig.canvas.mpl_connect('button_press_event', on_pick)
fig.canvas.mpl_connect('key_press_event', on_press)

def animate_func(i):
    global newGrid
    newGrid = mainLoop(newGrid)
    im.set_array(translateGrid(newGrid, BLOCK_VISABILITY))
    
    return [im]

anim = animation.FuncAnimation(fig,animate_func,frames = nSeconds * fps,interval = 1000 / fps, blit=False)
text = writeFPS()
dimtext = writeDIMENSION()
intensityText = writeINTENSITY()

tutorialText = writeTutorial()
plt.axis('off')
plt.show()
