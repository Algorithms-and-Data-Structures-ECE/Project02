import numpy as np
import math
import time

######################### Global Variables #########################
#my seed
np.random.seed(1063199)

#number of nodes/airports
NODE_NUMBER = 10

#The min and max weight for the node connections
LOWEST_WEIGHT = 50
HIGHEST_WEIGHT = 100

#number of trips
TRIPS_NUMBER = 5
#minimum weight for a trip
TRIP_WEIGHT = 200


######################### Initialisation and Printing Functions #########################
def createWeightsArray(nodeNumber):
    #create a random 2d array for the weights
    arr = np.random.randint(LOWEST_WEIGHT, HIGHEST_WEIGHT, size=(nodeNumber,nodeNumber))
    #The weight to go from one node to the same node is 0 (zero)
    np.fill_diagonal(arr, 0)
    return arr


def createTrip(nodeNumber,weightArr):
    #find the maximum length of a trip to get that many nodes
    lengthOfTrip = math.ceil(TRIP_WEIGHT / LOWEST_WEIGHT) + 1

    #temporary trip, (replace=False so that we don't choose the same node twice)
    a = np.random.choice(nodeNumber, lengthOfTrip, replace=False)

    #append the first 2 nodes since we need them to have at least 1 transition between nodes
    trip = [a[0], a[1]]

    #get the current trip weight
    curTripWeight = weightArr[a[0]][a[1]]
    
    #stop the trip after we exceed the weight
    for i in range(2, lengthOfTrip):
        trip.append(a[i])
        curTripWeight += weightArr[a[i-1]][a[i]]

        if curTripWeight > TRIP_WEIGHT:
            break

    return trip, curTripWeight


def printSparceMatrixData(jloc, jval, irow, next):
    #determine if we provided the simple or improved matrix
    matrixType = 1
    if next != []:
        matrixType = 2

    #These are present in both sparce Matrix
    print("Start of each trip (a{}_jloc) \n{}" .format(matrixType, jloc))
    print("Weight of each trip (a{}_jval) \n{}" .format(matrixType, jval))
    print("All the nodes (a{}_irow) \n{}" .format(matrixType, irow))

    #only the improved matrix has the next list
    if matrixType == 2:
        print("Next node index (a2_next), (-1 signals the end of the trip)\n{} \n" .format(next))
    else:
        print("")

    return

def printSparceMatrixTrips(jloc, irow, next):
    print("The trips more clearly are:")
    
    if next == []:
        for i in range(len(jloc)-1):
            print("Trip {} {}" .format(i, irow[jloc[i]:jloc[i+1]]))  

    else:         
        for i in range(len(jloc)-1):
            curNode = jloc[i]
            print("Trip {} [{}" .format(i, irow[curNode]) , end='')
            curNode = next[curNode]
            while curNode != -1:
                print(", {}" .format(irow[curNode]), end='')
                curNode = next[curNode]
            print("]")

    print("")
    return


def printBMatrixData(bRowStarts, bcol, bval, numOfZeroValues):
    print("b_iloc {}" .format(bRowStarts))
    print("b_jcol {}" .format(bcol))
    print("b_val {}" .format(bval))
    print("Number of zero elements we didn't store {}" .format(numOfZeroValues))
    print("")
    return

def printLeastMostVisitedNode(minPositions, min, maxPositions, max):
    print("The node", end="")
    if len(minPositions) > 1:
        print("s", end="")
    print(" with the least visits ", end="")
    if len(minPositions) > 1:
        print("are ", end="")
    else:
        print("is ", end="")

    print("{} with {} visit" .format(minPositions, min), end="")
    if min > 1:
        print("s", end="")
    print("")

    print("The node", end="")
    if len(maxPositions) > 1:
        print("s", end="")
    print(" with the most visits ", end="")
    if len(maxPositions) > 1:
        print("are ", end="")
    else:
        print("is ", end="")

    print("{} with {} visit" .format(maxPositions, max), end="")
    if max > 1:
        print("s", end="")
    print("\n")

    return


################################################## Simple Matrix ##################################################
def createSparceMatrixSimple(weightArr, tripsNumber, nodeNumber):
    #the index of the list nodes where the trip starts
    #(last element shows the end of the last trip)
    tripStarts = [0]
    #the total weight for the trip
    weightSums = []
    #all the nodes we pass in all the trips
    nodes = []
    
    for i in range(tripsNumber):
        trip, sum = createTrip(nodeNumber, weightArr)
        nodes.extend(trip)
        weightSums.append(sum)
        tripStarts.append(len(nodes))

    return tripStarts, weightSums, nodes

def addDestinationInSimpleMatrix(direction, destination, tripNumber, tripStarts, weightSums, nodes, weightArr):

    #the nodes before and after the start of the trip remain the same
    partStart = nodes[0:tripStarts[tripNumber]]
    partEnd = nodes[tripStarts[tripNumber+1]:]

    if direction == "start":  
        #first add the new destination then the rest of the trip
        partStart.append(destination)
        partStart.extend(nodes[tripStarts[tripNumber]:tripStarts[tripNumber+1]])
        #update the weight
        weightSums[tripNumber] += weightArr[destination][nodes[tripStarts[tripNumber]]]
    else:
        #first add the rest of the trip and then the destination
        partStart.extend(nodes[tripStarts[tripNumber]:tripStarts[tripNumber+1]])
        partStart.append(destination)
        #update the weight
        weightSums[tripNumber] += weightArr[nodes[tripStarts[tripNumber+1]-1]][destination]

    #join the lists to get the new nodes list
    nodes = partStart + partEnd

    #update all the following starts
    for i in range(tripNumber+1, len(tripStarts)):
        tripStarts[i] += 1

    return tripStarts, weightSums, nodes


################################################## Improved Matrix ##################################################
def createSparceMatrixImproved(weightArr, tripsNumber, nodeNumber):
    #we get the tripStarts,weightSums and nodes the same way
    tripStarts, weightSums, nodes = createSparceMatrixSimple(weightArr, tripsNumber, nodeNumber)

    #counter for which tripStart to use (starts at 1 instead of 0 to get the end of the first trip)
    counter = 1
    
    #to have a trip we need at least 2 nodes so by definition the first node points to the second
    nextNode = [1]

    #make it so if nextNode[i] == -1 then we have finished this trip
    for i in range(1, len(nodes)):
        if(tripStarts[counter] == i + 1):
            nextNode.append(-1)
            counter+=1
        else:
            nextNode.append(i + 1)

    return tripStarts, weightSums, nodes, nextNode


def addDestinationInImprovedMatrix(direction, destination, tripNumber, tripStarts, weightSums, nodes, nextNode, weightArr):
    #add the destination to the end of the nodes list
    nodes.append(destination)

    if direction == "start":
        #update the nextNode with start of the trip
        nextNode.append(tripStarts[tripNumber])

        #update the start of the trip
        tripStarts[tripNumber] = len(nextNode) - 1

        #update the weight
        weightSums[tripNumber] += weightArr[destination][nodes[tripStarts[tripNumber]]]

    else:
        #we use the variable last to store the last destination of the trip
        last = start = tripStarts[tripNumber]    
        next = nextNode[start]
    
        #when next == -1 it means that we reached the end of the trip
        while(next != -1):
            last = next
            next = nextNode[next]    
    
        #update the last element of the trip to point to the new destination
        nextNode[last] = len(nodes) - 1
        #add the end tag in the nextNode list for this new destination
        nextNode.append(-1)

        #update the weight
        weightSums[tripNumber] += weightArr[nodes[last]][destination]

    #the last tripStart shows the index of the last element
    tripStarts[-1] +=1

    return tripStarts, weightSums, nodes, nextNode


######################### Functions for Both Matrices #########################
def checkIfDestinationExistsInTrip(tripNumber, destination, tripStarts, nodes, nextNodes):
    if nextNodes == []:
        for i in range(tripStarts[tripNumber], tripStarts[tripNumber+1]):
            if destination == nodes[i]:
                return True

    else:
        curNode = tripStarts[tripNumber]
        while curNode != -1:
            if destination == nodes[curNode]:
                return True
            curNode = nextNodes[curNode]
            
    return False

def addDestination(direction, destination, tripNumber, tripStarts, weightSums, nodes, nextNode, weightArr):
    if checkIfDestinationExistsInTrip(tripNumber, destination, tripStarts, nodes, nextNode):
        print("The destination {} already exists in trip {}" .format(destination, tripNumber))

    else:
        if nextNode == []:
            tripStarts, weightSums, nodes = addDestinationInSimpleMatrix(
                direction, destination, tripNumber, tripStarts, weightSums, nodes, weightArr)
            print("Added destination {} at the {} of trip {} in Simple Matrix (A1)" .format(destination, direction, tripNumber))
        else:
            tripStarts, weightSums, nodes, nextNode = addDestinationInImprovedMatrix(
                direction, destination, tripNumber, tripStarts, weightSums, nodes, nextNode, weightArr)
            print("Added destination {} at the {} of trip {} in Improved Matrix (A2)" .format(destination, direction, tripNumber))
    return tripStarts, weightSums, nodes, nextNode


def findNumberOfMatches(tripA, tripB, tripStarts, nodes, nextNode):
    count = 0
    
    #this is for the simple sparce Matrix
    if nextNode == []:
        #if we are comparing a trip with itself then the number of matches is the number of nodes in the trip
        if tripA == tripB:
            return tripStarts[tripA+1] - tripStarts[tripA]
                #compare each node of trip A to all the nodes of trip B to find the number of matches
        for i in range(tripStarts[tripA], tripStarts[tripA+1]):
            for j in range (tripStarts[tripB], tripStarts[tripB+1]):
                if nodes[i] == nodes[j]:
                    count+=1

    #this is for the improved sparce Matrix
    else:
        #if we are comparing a trip with itself then the number of matches is the number of nodes in the trip
        if tripA == tripB:
            next = tripStarts[tripA]
            while next != -1:
                next = nextNode[next]
                count+=1
            return count

        #set the current node index as the start of trip
        curNodeA = tripStarts[tripA]
        
        #move node index until we reach -1 meaning that the trip has ended
        while curNodeA != -1:
            curNodeB = tripStarts[tripB]
            while curNodeB != -1:
                if nodes[curNodeA] == nodes[curNodeB]:
                    count+=1
                #set current node index the next node index
                curNodeB = nextNode[curNodeB]

            #set current node index the next node index
            curNodeA = nextNode[curNodeA]

    return count



def calculateB(tripStarts, nodes, nextNode):
    #the index of this list is the row
    #the value of it's row shows where the columns of this row start
    bRowStarts = [0]
    #this is where we store the columns which have a non zero value
    bcol = []
    #this where we store the actual value
    bval = []
    #this is just for testing purposes to see how much space we save
    numOfZeroValues = 0

    for i in range (len(tripStarts)-1):
        #the full 2d array is symmetrical we only need to check for the upper half (plus the diagonal)
        for j in range(i, len(tripStarts)-1):
            result = findNumberOfMatches(i, j, tripStarts, nodes, nextNode)
            if result != 0:
                bcol.append(j)
                bval.append(result)
            else:
                numOfZeroValues+=1

        #the next row will point to the end of the last column from the previous row
        bRowStarts.append(len(bcol))

    return bRowStarts, bcol, bval, numOfZeroValues



def findLeastMostVisitedNode(nodes, nodeNumber):
    #set all node visits to 0
    visits = [0] * nodeNumber

    #update the visit for each node
    for i in range(len(nodes)):
        visits[nodes[i]] +=1

    #(placeholder values) set the minimum and maximum as the position 0
    min = visits[0]
    minPositions = []
    max = visits[0]
    maxPositions = []

    #find the actual minimum value and the position(s)  
    for i in range(nodeNumber):
        if visits[i] < min:
            min = visits[i]
            minPositions.clear()
            minPositions = [i]
        elif visits[i] == min:
            minPositions.append(i)

        if visits[i] > max:
            max = visits[i]
            maxPositions.clear()
            maxPositions = [i]
        elif visits[i] == max:
            maxPositions.append(i)


    return minPositions, min, maxPositions, max


######################### Main Code #########################

print("-------------The weights array-------------\n")
weightArr = createWeightsArray(NODE_NUMBER)
print(weightArr)


print("\n---------------------------------------------------------------------------")
print("Simple Sparce Matrix (A1)\n")

a1_jloc, a1_jval, a1_irow = createSparceMatrixSimple(weightArr, TRIPS_NUMBER, NODE_NUMBER)
printSparceMatrixData(a1_jloc, a1_jval, a1_irow, [])
printSparceMatrixTrips(a1_jloc, a1_irow, [])


print("\n---------------------------------------------------------------------------")
print("Improved Sparce Matrix (A2)\n")

a2_jloc, a2_jval, a2_irow, a2_next = createSparceMatrixImproved(weightArr, TRIPS_NUMBER, NODE_NUMBER)
printSparceMatrixData(a2_jloc, a2_jval, a2_irow, a2_next)
printSparceMatrixTrips(a2_jloc, a2_irow, a2_next)

print("\n---------------------------------------------------------------------------")
print("Adding Destinations\n")



#add destination 0 at the start of trip 0 in matrix A1
a1_jloc, a1_jval, a1_irow, _ = addDestination("start", 0, 0, a1_jloc, a1_jval, a1_irow, [], weightArr)
#add destination 6 at the end of trip 0 in matrix A1
a1_jloc, a1_jval, a1_irow, _ = addDestination("end", 6, 0, a1_jloc, a1_jval, a1_irow, [], weightArr)
#add destination 2 at the start of trip 0 in matrix A1
a1_jloc, a1_jval, a1_irow, _ = addDestination("start", 2, 0, a1_jloc, a1_jval, a1_irow, [], weightArr)
printSparceMatrixData(a1_jloc, a1_jval, a1_irow, [])
printSparceMatrixTrips(a1_jloc, a1_irow, [])

#add destination 0 at the start of trip 0 in matrix A2
a2_jloc, a2_jval, a2_irow, a2_next = addDestination("start", 8, 1, a2_jloc, a2_jval, a2_irow, a2_next, weightArr)
#add destination 3 at the end of trip 1 in matrix A2
a2_jloc, a2_jval, a2_irow, a2_next = addDestination("end", 3, 1, a2_jloc, a2_jval, a2_irow, a2_next, weightArr)
#add destination 8 at the start of trip 1 in matrix A2
a2_jloc, a2_jval, a2_irow, a2_next = addDestination("start", 4, 1, a2_jloc, a2_jval, a2_irow, a2_next, weightArr)
printSparceMatrixData(a2_jloc, a2_jval, a2_irow, a2_next)
printSparceMatrixTrips(a2_jloc, a2_irow, a2_next)



print("\n---------------------------------------------------------------------------")
print("Calculating B (A1 Matrix)\n")
bRowStarts, bcol, bval, numOfZeroValues = calculateB(a1_jloc, a1_irow, [])
printBMatrixData(bRowStarts, bcol, bval, numOfZeroValues)

print("Calculating B (A2 Matrix)\n")
bRowStarts, bcol, bval, numOfZeroValues = calculateB(a2_jloc, a2_irow, a2_next)
printBMatrixData(bRowStarts, bcol, bval, numOfZeroValues)


print("\n---------------------------------------------------------------------------")
print("Finding most and least visited nodes (A1 Matrix)\n")
minPositions, min, maxPositions, max = findLeastMostVisitedNode(a1_irow, NODE_NUMBER)
printLeastMostVisitedNode(minPositions, min, maxPositions, max)

print("Finding most and least visited nodes (A2 Matrix)\n")
minPositions, min, maxPositions, max = findLeastMostVisitedNode(a2_irow, NODE_NUMBER)
printLeastMostVisitedNode(minPositions, min, maxPositions, max)


'''

#For 1.000 trips
weightArr = createWeightsArray(800)
a1_jloc, a1_jval, a1_irow = createSparceMatrixSimple(weightArr, 1000, 800)
a2_jloc, a2_jval, a2_irow, a2_next = createSparceMatrixImproved(weightArr, 1000, 800)


print("Calculating B (A1 Matrix 1000 elements)")
startTime = time.time()
bRowStarts, bcol, bval, numOfZeroValues = calculateB(a1_jloc, a1_irow, [])
endTime = time.time()
totalTime = endTime - startTime
print("Total time to calculate B (A1 Matrix 1000 elements): {} seconds" .format(totalTime))

print("Calculating B (A2 Matrix 1000 elements)")
startTime = time.time()
bRowStarts, bcol, bval, numOfZeroValues = calculateB(a2_jloc, a2_irow, a2_next)
endTime = time.time()
totalTime = endTime - startTime
print("Total time to calculate B (A2 Matrix 1000 elements): {} seconds\n" .format(totalTime))

print("Finding most and least visited nodes (A1 Matrix 1000 elements)")
startTime = time.time()
minPositions, min, maxPositions, max = findLeastMostVisitedNode(a1_irow, 800)
endTime = time.time()
totalTime = endTime - startTime
print("Total time to find most and least visited nodes (A1 Matrix 1000 elements): {} seconds" .format(totalTime))
printLeastMostVisitedNode(minPositions, min, maxPositions, max)

print("Finding most and least visited nodes (A2 Matrix 1000 elements)")
startTime = time.time()
minPositions, min, maxPositions, max = findLeastMostVisitedNode(a2_irow, 800)
endTime = time.time()
totalTime = endTime - startTime
print("Total time to find most and least visited nodes (A2 Matrix 1000 elements): {} seconds" .format(totalTime))
printLeastMostVisitedNode(minPositions, min, maxPositions, max)



#For 2.000 trips
weightArr = createWeightsArray(800)
a1_jloc, a1_jval, a1_irow = createSparceMatrixSimple(weightArr, 2000, 800)
a2_jloc, a2_jval, a2_irow, a2_next = createSparceMatrixImproved(weightArr, 2000, 800)


print("Calculating B (A1 Matrix 2000 elements)")
startTime = time.time()
bRowStarts, bcol, bval, numOfZeroValues = calculateB(a1_jloc, a1_irow, [])
endTime = time.time()
totalTime = endTime - startTime
print("Total time to calculate B (A1 Matrix 2000 elements): {} seconds" .format(totalTime))

print("Calculating B (A2 Matrix 2000 elements)")
startTime = time.time()
bRowStarts, bcol, bval, numOfZeroValues = calculateB(a2_jloc, a2_irow, a2_next)
endTime = time.time()
totalTime = endTime - startTime
print("Total time to calculate B (A2 Matrix 2000 elements): {} seconds\n" .format(totalTime))

print("Finding most and least visited nodes (A1 Matrix 2000 elements)")
startTime = time.time()
minPositions, min, maxPositions, max = findLeastMostVisitedNode(a1_irow, 800)
endTime = time.time()
totalTime = endTime - startTime
print("Total time to find most and least visited nodes (A1 Matrix 2000 elements): {} seconds" .format(totalTime))
printLeastMostVisitedNode(minPositions, min, maxPositions, max)

print("Finding most and least visited nodes (A2 Matrix 2000 elements)")
startTime = time.time()
minPositions, min, maxPositions, max = findLeastMostVisitedNode(a2_irow, 800)
endTime = time.time()
totalTime = endTime - startTime
print("Total time to find most and least visited nodes (A2 Matrix 2000 elements): {} seconds" .format(totalTime))
printLeastMostVisitedNode(minPositions, min, maxPositions, max)


'''