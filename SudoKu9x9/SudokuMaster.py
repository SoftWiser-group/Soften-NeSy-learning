import math
from random import randint
CONST_SIZE = 9 #Please make sure that the size is a perfect square number


def getColoumn(index, arr2d):
    coloumn = []
    for row in arr2d:
        coloumn.append(row[index])
    return coloumn

def getInnerMatrix(rowIndex, colIndex, arr2d):
    innerMatrix = []
    sizeOfInnerMatrix = int(math.sqrt(CONST_SIZE))
    startRowIndex = 0
    startColIndex = 0
    while((startRowIndex + sizeOfInnerMatrix) <= rowIndex):
        startRowIndex+=sizeOfInnerMatrix
    while((startColIndex + sizeOfInnerMatrix) <= colIndex):
        startColIndex+=sizeOfInnerMatrix
    endRowIndex = startRowIndex + sizeOfInnerMatrix
    endColIndex = startColIndex + sizeOfInnerMatrix
    for i in range(startRowIndex, endRowIndex):
        for j in range(startColIndex, endColIndex):
            innerMatrix.append(arr2d[i][j])
    return innerMatrix

def checkBoardValidity(arr2d):
    for i in range(CONST_SIZE):
        row = arr2d[i]
        for j in range(CONST_SIZE):
            ele = row[j]
            if (row.count(ele) != 1):
                return False
            col = getColoumn(j, arr2d)
            if (col.count(ele) != 1):
                return False
            innerMatrix = getInnerMatrix(i, j, arr2d)
            if (innerMatrix.count(ele) != 1):
                return False
    return True

def isBoardEmpty(arr2d):
    for row in arr2d:
        for ele in row:
            if ele == 0:
                return True
    return False

def find_empty_location(arr,l):
	for row in range(9):
		for col in range(9):
			if(arr[row][col]==0):
				l[0]=row
				l[1]=col
				return True
	return False
    
def solveBoard(arr2d):
    l=[0,0]
    if(not find_empty_location(arr2d, l)):
        return True
    row=l[0]
    col=l[1]
    for num in range(1,CONST_SIZE+1):
        safeList = getAllPossibleNumbersInPlace(row, col, arr2d)
        if num in safeList:
            arr2d[row][col] = num
            if(solveBoard(arr2d)):
                return True
            arr2d[row][col] = 0
    return False


def shuffleBoard(arr2d):
    chooseNumber = -1
    replacingNumber = -1
    while(replacingNumber == chooseNumber):
        chooseNumber = randint(1, CONST_SIZE)
        replacingNumber = randint(1, CONST_SIZE)
    for i in range(0, CONST_SIZE):
        for j in range(0, CONST_SIZE):
            if(arr2d[i][j] == chooseNumber):
                arr2d[i][j] = replacingNumber
            elif(arr2d[i][j] == replacingNumber):
                arr2d[i][j] = chooseNumber

    sizeOfInnerMatrix = int(math.sqrt(CONST_SIZE))
    if (sizeOfInnerMatrix > 1):
        chooseRowIndex = -1
        replacingRowIndex = -1
        while(chooseRowIndex == replacingRowIndex):
            chooseRowIndex = randint(1, sizeOfInnerMatrix)
            replacingRowIndex = randint(1, sizeOfInnerMatrix)
        multiplier = randint(0, sizeOfInnerMatrix-1)
        chooseRowIndex += (multiplier*sizeOfInnerMatrix)
        replacingRowIndex += (multiplier*sizeOfInnerMatrix)
        arr2d[chooseRowIndex - 1], arr2d[replacingRowIndex - 1] = arr2d[replacingRowIndex -1], arr2d[chooseRowIndex - 1]
    return arr2d

def getAllPossibleNumbersInPlace(rowIndex, colIndex, arr2d):
    row = arr2d[rowIndex]
    col = getColoumn(colIndex, arr2d)
    innerMatrix = getInnerMatrix(rowIndex, colIndex, arr2d)
    posibilities = [x for x in range(1, CONST_SIZE+1) if ((x not in row) and (x not in col) and (x not in innerMatrix))]
    
    return posibilities

def removeLogically(arr2d, cutOff=35):
    removedItems = 0
    for _ in range(CONST_SIZE*500):
        i = randint(0, CONST_SIZE-1)
        j = randint(0, CONST_SIZE-1)
        temp = arr2d[i][j]
        if(temp == 0):
            continue
        arr2d[i][j] = 0
        if(len(getAllPossibleNumbersInPlace(i, j, arr2d)) != 1):
            arr2d[i][j] = temp
        else:
            removedItems+=1
        if(removedItems == cutOff):
            return

def removeRandomly(board, cutOff):
    removedItem = 0
    for i in range(CONST_SIZE):
        for j in range(CONST_SIZE):
            if(board[i][j] == 0):
                continue
            temp = board[i][j]
            board[i][j] = 0
            tempBoard = [[ele for ele in row] for row in board]
            if(not solveBoard(tempBoard)):
                board[i][j] = temp
            else:
                removedItem += 1
            if(removedItem == cutOff):
                return board
    return board
            

def printBoard(arr2d):
    for row in arr2d:
        print(row)

def makeBoard():
    # board = [[0 for _ in range(CONST_SIZE)] for _ in range(CONST_SIZE)]
    # for i in range(0, CONST_SIZE):
        # for j in range(0, CONST_SIZE):
            # board[i][j] = int((i * math.sqrt(CONST_SIZE) + int(i / math.sqrt(CONST_SIZE)) + j) % CONST_SIZE) + 1
    i = randint(0, 4)
    if i == 0:
        board = [[1, 2, 3, 4, 5, 6, 7, 8, 9], [4, 5, 6, 7, 8, 9, 1, 2, 3], [7, 8, 9, 1, 2, 3, 4, 5, 6], \
                 [2, 3, 4, 5, 6, 7, 8, 9, 1], [5, 6, 7, 8, 9, 1, 2, 3, 4], [8, 9, 1, 2, 3, 4, 5, 6, 7], \
                 [3, 4, 5, 6, 7, 8, 9, 1, 2], [6, 7, 8, 9, 1, 2, 3, 4, 5], [9, 1, 2, 3, 4, 5, 6, 7, 8]]
    elif i == 1:
        board = [[1, 2, 3, 4, 5, 6, 7, 8, 9], [6, 7, 8, 9, 1, 2, 3, 4, 5], [5, 4, 9, 8, 7, 3, 2, 1, 6], \
                 [4, 8, 6, 3, 9, 1, 5, 7, 2], [2, 1, 7, 6, 4, 5, 8, 9, 3], [3, 9, 5, 7, 2, 8, 1, 6, 4], \
                 [7, 5, 2, 1, 6, 9, 4, 3, 8], [9, 3, 1, 2, 8, 4, 6, 5, 7], [8, 6, 4, 5, 3, 7, 9, 2, 1]]
    elif i == 2:
        board = [[1, 2, 3, 4, 5, 6, 7, 8, 9], [4, 8, 6, 2, 7, 9, 5, 1, 3], [9, 5, 7, 1, 3, 8, 2, 6, 4], \
                 [2, 3, 4, 6, 8, 5, 1, 9, 7], [6, 7, 8, 9, 1, 2, 3, 4, 5], [5, 9, 1, 7, 4, 3, 8, 2, 6], \
                 [7, 4, 9, 3, 2, 1, 6, 5, 8], [8, 6, 2, 5, 9, 7, 4, 3, 1], [3, 1, 5, 8, 6, 4, 9, 7, 2]]
    elif i == 3:
        board = [[1, 2, 3, 4, 5, 6, 7, 8, 9], [6, 4, 5, 9, 7, 8, 3, 1, 2], [7, 9, 8, 2, 1, 3, 5, 4, 6], \
                 [9, 6, 7, 8, 4, 2, 1, 3, 5], [5, 1, 4, 3, 9, 7, 2, 6, 8], [3, 8, 2, 5, 6, 1, 4, 9, 7], \
                 [2, 3, 9, 7, 8, 4, 6, 5, 1], [4, 5, 1, 6, 2, 9, 8, 7, 3], [8, 7, 6, 1, 3, 5, 9, 2, 4]]
    elif i == 4:
        board = [[1, 2, 3, 4, 5, 6, 7, 8, 9], [8, 7, 6, 2, 9, 3, 1, 4, 5], [4, 9, 5, 7, 8, 1, 3, 2, 6], \
                 [2, 3, 9, 6, 1, 5, 4, 7, 8], [7, 6, 1, 8, 4, 9, 2, 5, 3], [5, 8, 4, 3, 7, 2, 6, 9, 1], \
                 [6, 4, 7, 9, 3, 8, 5, 1, 2], [9, 1, 2, 5, 6, 4, 8, 3, 7], [3, 5, 8, 1, 2, 7, 9, 6, 4]]

    randomInt = randint(8, 200)
    for _ in range(randomInt):
        board = shuffleBoard(board)

    return board


def makePuzzleBoard(board, level="easy"):
    sizeSquare = CONST_SIZE*CONST_SIZE
    levels = {
        "easy" : ((int(sizeSquare/2) - int(sizeSquare/10)), 0),
        "moderate" : (int(sizeSquare), int(sizeSquare/15)),
        "difficult" : (int(sizeSquare), int(sizeSquare/10))
        }
    logicalCutOff = levels[level][0]
    randomCutOff = levels[level][1]
    removeLogically(board, logicalCutOff)
    if randomCutOff != 0:
        removeRandomly(board, randomCutOff)
    return board

"""
board = makeBoard()
puzzle = makePuzzleBoard(board, "moderate")
printBoard(puzzle)
#puzzle is a 2-d array, try print(puzzle)
if solveBoard(puzzle):
    print("\n\n\nSolved Solution is: ")
    printBoard(puzzle)
else:
    print("No Solution Exist")
    """