# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.shortcuts import render

import sys
import random
import time
import itertools
import urllib
import csv
import copy
import functools
import math
from itertools import permutations
from django.shortcuts import render, redirect
from django.http import JsonResponse
from django.conf import settings
from django.views.decorators.csrf import csrf_exempt

from .models import scoreboard
from .forms import registerForm

from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth.models import User
from django.contrib.sites.models import Site
from registration.models import RegistrationProfile



# Create your views here.

def home(request):
	context = {}
	MYSITE = Site.objects.get_current()
	MYSITE.domain = '127.0.0.1:8000'
	MYSITE.name = 'My Site'
	MYSITE.save()

	if request.user.is_authenticated():
		queryset = User.objects.all().order_by('-timestamp')
		context = {
			"queryset": queryset,
		}

	context = {}

	return render(request, 'landingpage.html', context)

# landing page
@csrf_exempt
def landingpageTSP(request):
	tsp1 = request.POST.get('tsp1', None)
	tsp1 = tsp1.split(',')
	tsp1 = map(float, tsp1)
	result = landingBackbones(tsp1)

	context = {
		'tour': result[0],
		'backbones': result[3],
	}
	return JsonResponse(context)

# results page
def vsliresults(request):
	context = {

	}

	return render(request, 'vsli.html', context)

# Create your views here.
def graphOutput(request):
	#this takes a request from the front end and then the response is simple to render the html templates
	context = {}

	return render(request, 'plot.html', context)


def getATour(request):

	citiesNum = request.GET.get('numCities', 10)
	backboneLen = request.GET.get('backboneLen', 0)
	# Seed can be any number between 0 - 40 even floats
	newInstance = request.GET.get('newInstance', 0)

	if int(citiesNum) > 1000:
		print "Instance size of {:} is too large, size set to 1000.".format(citiesNum)
		citiesNum = 1000

	#this takes a request from the front end and then the response is simple to render the html template
	try:
		tourBB = info(Cities(int(citiesNum),
							 seed=float(newInstance)),
							 int(backboneLen))
	except:
		tourBB = info(Cities(int(citiesNum),
							 seed=float(newInstance)))

	context = {
		"tour": tourBB[0],
		"bbData": tourBB[1],
		"tourBBLen": tourBB[2],
		"backbones": tourBB[3],
		"numTours": citiesNum,
	}
	return JsonResponse(context)

def userTour(request):
	pathLen = request.GET.get('pathLen', None)
	tourVal = request.GET.get('tourVal', None)
	tourKey = str((request.GET.get('tourKey', None)))
	tourBBLen = request.GET.get('tourBBLen', None)

	print pathLen
	print tourVal
	print tourKey
	print tourBBLen

	# convert tour key into list of cities user visited
	tourKey = tourKey[slice(15, len(tourKey) - 7)].split(',')
	tourKeyArr = [int(x) for x in tourKey]

	#get the list of cities
	tourVal = tourVal.split(',')
	tourVal = map(float, tourVal)

	benchmarkBB(tourVal)

	#define variables for context
	tour = []
	tourLen = 0
	valid = False
	ratios = " "

	if len(tourKeyArr) == int(pathLen):
	 	valid = True

	#get the tour
	for i in tourKeyArr:
		tour.append(tourVal[i*2])
		tour.append(tourVal[i*2 + 1])

	#get the tour length
	if valid == True:
		tourLen = "Tour distance: {:.2f}".format(calcDist(tour))
		ratios = ratio(float(tourBBLen), calcDist(tour))
	else:
		tourLen = "Invalid Tour of length " + str(len(tourKeyArr))

	context = {
		"tour": tour,
		"valid": valid,
		"tourlen": tourLen,
		"ratios": ratios,
	}

	return JsonResponse(context)

def googCTSP(request):
	tour = request.GET.get('tourl', None)
	# potential bug is tour is given with [] but it seems to not be here
	tour = tour.split(',')
	tourArr = [float(x) for x in tour]
	tourCities = set()

	for i in range(len(tourArr)/2):
		tourCities.add(City(tourArr[i*2], tourArr[i*2+1]))
	

	tourBB = info(tourCities)

	context = {
		"tour": tourBB[0],
		"bbData": tourBB[1],
		"tourBBLen": tourBB[2],
		"backbones": tourBB[3],
		"lentour": len(tourArr)/2,
	}

	return JsonResponse(context)

def calcDist(strTour):
	dist = 0
	for i in range(0, len(strTour)/2):
		if i == (len(strTour)/2 - 1):
			dist = dist + math.sqrt(abs(strTour[(i*2)] - strTour[0])**2 +
									abs(strTour[(i*2) + 1] - strTour[1])**2)
		else:
			dist = dist + math.sqrt(abs(strTour[(i*2)] - strTour[(i*2)+2])**2 +
									abs(strTour[(2*i)+1] - strTour[(2*i)+3])**2)
	return dist

def submitTour(request):
	usrPath = request.GET.get('tourKey', None)
	bbLen = request.GET.get('tourBBLen', None)
	tsp = request.GET.get('tsp', None)
	userLength = request.GET.get('usrLen', None)
	usrPathLen = request.GET.get('pathLen', None)
	numNodes = request.GET.get('input1', None)

	tuple = {}

	if request.user.is_authenticated():
		if int(usrPathLen) == int(numNodes) and userLength != '0':
			tsp = tsp.split(',')
			tsp = map(float, tsp)


			tspKV = list()
			for i in range(0, int(numNodes)):
				tspKV.append((tsp[i*2], tsp[i*2+1], i))



			usrPath = str(usrPath)
			usrPath = usrPath[15:len(usrPath) - 7]

			newTupleScoreboard = scoreboard()
			newTupleScoreboard.id = request.user
			newTupleScoreboard.name = request.user.first_name + ' ' + request.user.last_name
			newTupleScoreboard.tourProblem = str(tspKV)
			newTupleScoreboard.userSolution = str(usrPath)
			newTupleScoreboard.userLength = userLength
			newTupleScoreboard.algorithmName = "Backbone Algorithm"
			newTupleScoreboard.algorithmLength = str(round(float(bbLen),2))
			newTupleScoreboard.numberOfNodes = numNodes
			newTupleScoreboard.save()
			return JsonResponse(tuple)


def gethiscores(request):
	temp = list()
	queryset = scoreboard.objects.all()

	for i in range(len(queryset)):
		temp2 = list()
		temp2.append(str(queryset[i].id))
		temp2.append(queryset[i].name)
		temp2.append(queryset[i].tourProblem)
		temp2.append(queryset[i].userSolution)
		temp2.append(queryset[i].userLength)
		temp2.append(queryset[i].algorithmName)
		temp2.append(queryset[i].algorithmLength)
		temp2.append(queryset[i].dateSubmitted)
		temp2.append(queryset[i].numberOfNodes)
		temp.append(temp2)

	context = {
		"queryset": temp
	}
	return JsonResponse(context)


def register(request):
	args = {}
	if request.method == 'POST':
		form = registerForm(request.POST)
		if form.is_valid():
			mysite = Site.objects.get_current()
			usr = form.save()
			regProfile = RegistrationProfile.objects.create_profile(usr)
			regProfile.send_activation_email(mysite)
		return redirect('http://127.0.0.1:8000', args)
	else:
		form = registerForm()
		args = {'form': form}
		return render(request, 'registration/registration_form.html' , args)

# Backend code

# Contains code for a brute force approach and a heuristic, the nearest-neighbor

#--------------------------------------------------------
#Brute Force Approach

#generate all possible tours given a set of cities
possibletours = itertools.permutations

#gets the shortest tour from a set of cities
def alltours(cities):
	return shortestTour(possibletours(cities))

def shortestTour(tours):
	return min(tours, key=tourLength)

#--------------------------------------------------------

#returns the length of a tour
def tourLength(tour):
	return sum(distance(tour[i-1], tour[i]) for i in range(0, len(tour)))

#representation of a city location
class Point(complex):
	x = property(lambda p: p.real)
	y = property(lambda p: p.imag)

#a city can be initialised as a point
City = Point


#distance between two point where one or both could be a backbone
def distance(A, B, *args):
	if str(type(B)) == "<class 'bbn.views.Backbone'>" and str(type(A)) == "<class 'bbn.views.Backbone'>":
		tmp = min(abs(A.y[0] - B.y[0]), abs(A.y[-1] - B.y[0]), abs(A.y[0] - B.y[-1]), abs(A.y[-1] - B.y[-1]))
		if abs(A.y[0] - B.y[0]) == tmp:
			A.z = 'L'
			B.z = 'L'
		elif abs(A.y[-1] - B.y[0]) == tmp:
			A.z = 'R'
			B.z = 'L'
		elif abs(A.y[0] - B.y[-1]) == tmp:
			A.z = 'L'
			B.z = 'R'
		elif abs(A.y[-1] - B.y[-1]) == tmp:
			A.z = 'R'
			B.z = 'R'
		return tmp


	elif str(type(A)) == "<class 'bbn.views.Backbone'>":
		if A.z == 'R':
			return abs(A.y[0] - B)
		elif A.z == 'L':
			return abs(A.y[-1] - B)
		elif abs(A.y[0] - B) < abs(A.y[-1] - B):
			A.z = 'Lunsure'
			return abs(A.y[0] - B)
		elif abs(A.y[0] - B) >= abs(A.y[-1] - B):
			A.z = 'Runsure'
			return abs(A.y[-1] - B)

	elif str(type(B)) == "<class 'bbn.views.Backbone'>":
		if B.z == 'R':
			return abs(A - B.y[0])
		elif B.z == 'L':
			return abs(A - B.y[-1])
		elif abs(A - B.y[0]) < abs(A - B.y[-1]):
			B.z = 'Lunsure'
			return abs(A - B.y[0])
		elif abs(A - B.y[0]) >= abs(A - B.y[-1]):
			B.z = 'Runsure'
			return abs(A - B.y[-1])
	return abs(A - B)

#return the first element
def firstElem(collection):
	return next(iter(collection))

#nearest neighbor from a cities to city and checks if nearest neighbor is a backbone city
def nearest_neighbor(city, cities):
	minDist = min(cities, key=lambda b:distance(b, city))
	if str(type(minDist)) == "<class 'bbn.views.Backbone'>":
		if minDist.z == 'Lunsure':
			minDist.z = 'L'
		elif minDist.z == 'R':
			minDist.z = 'R'
	return minDist

#nearest neighbor starts tour at arbitrary city in the tour
def nnVariation(cities, start=None):
	if start is None:
		start = firstElem(cities)

	tour = [start]
	unvisited = set(cities - {start})
	while unvisited:
		candidate = nearest_neighbor(tour[-1], unvisited)
		tour.append(candidate)
		unvisited.remove(candidate)

	#cleans backbones of weather it has been used
	for city in tour:
	    if str(type(city)) == "<class 'bbn.views.Backbone'>":
	        city.z = ' '

	return tour

#makes a set of cities with in a rectangle, there are 50 possible seeds thus sets you can have
def Cities(n, width=1000, height=600, seed=50):
	random.seed(seed*n)
	tempSet = set()
	for a in range(n):
		tempCity = City(random.randrange(width), random.randrange(height))
		while tempCity in tempSet:
			print "Finding a unique city ..."
			tempCity = City(random.randrange(width), random.randrange(height))
		tempSet.add(tempCity)
	return frozenset(tempSet)

#put in a tour and connects the last and first point
def plotTour(tour, *args):
	if not args:
		plotPoints(list(tour) + [tour[0]])
	else:
		plotPoints(list(tour))


#takes a list of points and plots
def plotPoints(points):
	x = []
	y = []

	#checks if a point is a backbone, will plot average point
	for p in points:
		if str(type(p)) == "<class 'bbn.views.Backbone'>":
			x.append(p.x.x)
			y.append(p.x.y)
		else:
			x.append(p.x)
			y.append(p.y)
	plt.plot(x, y, 'bo-')

#apply a tsp algo to cities
def applyTSP(algo, cities):
	t0 = time.clock()
	tour = algo(cities)
	t1 = time.clock()
	print("{} city tour with length {:.1f} in {:.3f} secs for {}".format(len(tour), tourLength(tour), t1 - t0, algo.__name__))
	plotTour(tour)
	plt.show()

#applyTSP(nnVariation, Cities(4, 4))

#ANALYSIS length between two algorithms
def ratio(cities, cities2):
	if (cities - cities2) == 0:
		return "Your tour is {:.1f} % better than the BB Tour".format(0)
	elif (cities - cities2) <= 0:
		return "Your tour is {:.3f}% worse than the BB Tour.".format((abs(cities - cities2)/cities))
	elif (cities - cities2) > 0:
		return "your tour is {:.3f}% better than the BB Tour".format((abs(cities - cities2)/cities))
	else:
		return "Invalid Tour Input"

# sampled approach to nnVariation
def repeatedNN(cities, repititions = 100):
	return shortestTour(nnVariation(cities, start) for start in sample(cities, repititions))

def repeatedNN10(cities):
	return repeatedNN(cities, 10)

def repeatedNN100(cities):
	return repeatedNN(cities, 100)

def sample(cities, repititions, seed=42):
	return random.sample(cities, repititions)

#print repeatedNN(Cities(20, 4), 10)

#Tuple of maps, the maps are of cities Maps(30, 60), 30 maps each containing 60 cities and unique to numCities value
def Maps(numMaps, numCities):
	return tuple(Cities(numCities, seed=(n, numCities))
				 for n in range(numMaps))

#print Maps(4, 4)

#ANALYSIS average time for all inputs
def benchmark(funct, cities):
	t0 = time.clock()
	results = [funct(x) for x in cities]
	t1 = time.clock()
	return (t1 - t0/len(cities), results)

#ANALYSIS for many tsp algorithms
def benchmarks(functions, maps=Maps(30,60)):
	for function in functions:
		t, result = benchmark(function, maps)
		length = [tourLength(r) for r in result]
		print("{:>25} |{:7.0f} ±{:4.0f} ({:5.0f} to {:5.0f}) |{:7.3f} secs/map | {} ⨉ {}-city maps"
              .format(function.__name__, mean(lengths), stdev(lengths), min(lengths), max(lengths),
                      t, len(maps), len(maps[0])))

#algorithms = [nn_tsp, repeat_10_nn_tsp, repeat_25_nn_tsp, repeat_50_nn_tsp, repeat_100_nn_tsp]

#----------------------


#backbone class x is complex number, y is the backbone, z is connectors
class Backbone(object):
	def __init__(self, x, y, *args):
		self.x = x
		self.y = y
		self.z = dict(args)

	def __repr__(self):
		return repr(self.x)


#this gets ALL common backbone between a set of tours of arbitrary lengths, ommitting sub-backbones of long bones
def getAllBackBones(tours, *args):
	#number of tours
	lengthTours = len(tours)

	#get first tour and its length
	firstTour = tours[0]
	lengthFirst = len(firstTour)

	#store backbones
	backbones = list()

	switch = False

	#all combinations
	for i in range(0, lengthFirst):
		for j in range(i + 1, lengthFirst + 1):
			tmp = firstTour[i:j]
			#loops through each tour to check if tmp is contained
			k = 1
			for k in range(k, lengthTours):
				tmpStr = ''.join(str(e) for e in tours[k])
				tmpTmp = ''.join(str(e) for e in tmp)
				#if subtour is not in any of the tours then we append the current bb
				if not (tmpTmp in tmpStr):
					if switch:
						backbones.append(firstTour[i:j - 1])
						switch = False
					break
				#we get to the end of the check of tours then we hit a inital match!
				if k==lengthTours-1 and not switch:
					switch = True

	if len(args[0]) == 0:
		return prunebackBonesGELE(backbones, 2, 3)
	else:
		for x in args:
			return prunebackBonesLE(backbones, x[0])

#takes a list of backbones and returns a backbone less than or equal too
#optimising backbones
def prunebackBonesLE(backbones, length):
	prunedBackbones = list()
	for backbone in backbones:
		if len(backbone) <= length:

			prunedBackbones.append(backbone)
	return prunedBackbones

def prunebackBonesGE(backbones, length):
	prunedBackbones = list()
	for backbone in backbones:
		if len(backbone) >= length:
			prunedBackbones.append(backbone)
	return prunedBackbones

#backbones inbetween len1 and len2 or outside
def prunebackBonesGELE(backbones, length1, length2):
	prunedBackbones = list()
	for backbone in backbones:
		if len(backbone) >= length1 and len(backbone) <= length2:
			prunedBackbones.append(backbone)
	return prunedBackbones

def prunebackBonesLEGE(backbones, length1, length2):
	prunedBackbones = list()
	for backbone in backbones:
		if len(backbone) <= length1 and len(backbone) >= length2:
			prunedBackbones.append(backbone)
	return prunedBackbones

#convert a subset of tours into a backbone
def bbEdgesToBB(backboneEdges):
	real = []
	img = []

	for city in backboneEdges:
		real.append(city.x)
		img.append(city.y)

	return Backbone(City(round(sum(real)/len(backboneEdges),0), round(sum(img)/len(backboneEdges),0)) , backboneEdges)

#substitute the backbone into the set of tours generated by nnVariation
def bbSubstitution(collection, backbone):
	tours = list()
	for path in collection:
		tmp = copy.deepcopy(path)
		num = 0
		for i in range(0, len(backbone.y)):
			if i == 0:
				num = tmp.index(backbone.y[0])
				tmp[num] = backbone
			else:
				tmp.remove(backbone.y[i])
		tours.append(tmp)
	return tours

#applies the backbone onto xyz; is is the only time we really use KNN (to generate tours). We just feed it a tour
def bbReduction(tour, *args):
	tspSolutions = list()

	#one or zero length tours
	if len(tour) < 2:
		return tour

	#get a set of good tours
	for city in tour:
		tspSolutions.append(nnVariation(tour, city))

	#solutions into a dictionary by their length
	topDict = {}
	meanSqu = {}
	# Holds the set of tours we examine for backbones
	top = list()
	avg = 0
	standardDev = 0

	# calculates the length for each tour
	for tours in tspSolutions:
		if type(tours) == 'Point' or len(tours) < 2:
			topDict[tuple(tours)] = -1
		else:
			topDict[tuple(tours)] = tourLength(tours)

	# Calculate the average of the tours in dictionary
	avg = sum(topDict.values())/ len(topDict)

	# Calculate the standard deviation
	for val in topDict:
		meanSqu[val] = (topDict[val] - avg)**2
	standardDev = math.sqrt(sum(meanSqu.values())/ len(meanSqu))

	# Store the distribution of lengths
	store = {}
	# Number of samples for the distribution
	k = 0

	candidate = 999999

    # a Tour len < 500 cannot generate significant statistical data so a differ-
	# stategy is used
	if(len(tour) < 500):
		# gets top 10 percent of shortest tours
		for i in range(0, int(round(len(tspSolutions)*0.6 + 0.5))):
            #picking method
			tmp = random.choice(topDict.keys())
			#tmp = max(topDict, key=topDict.get)
            #tmp = min(topDict, key=topDict.get)
			top.append(list(tmp))
			del topDict[tmp]
	else:
		while(candidate > (avg + (standardDev/2)) and k < 200):
            #a random list of tours from dictionary of tours and calculates average
			top = list()
			ranSample = random.sample(topDict, int(round(len(tspSolutions)*0.1 + 0.5)))
			avgRanSamp = 0
			for item in ranSample:
				avgRanSamp += topDict[item]
				top.append(list(item))
			candidate = avgRanSamp / len(ranSample)
            #store these averages in a dictionary
			if candidate < avg1:
				try:
					store[round(candidate, 2)].append(top)
					store[round(candidate, 2)][0] = store[round(candidate, 2)][0] + 1
				except:
					store[round(candidate, 2)] = [1, top]
				candidate = 999999999
			k += 1
			top = list()
        #picking most common tour lengths from dictionary method
		maxLen = 0
		for item in store:
			if store[item][0] > maxLen and store[item][0] > 1:
				maxLen = store[item][0]
				top = store[item][random.randint(1, len(store[item]) - 1)]

	# looks for BB in all the tours in top
	# substitute these backbones into all the tours
	reducedTours = set()
	for backbone in getAllBackBones(top, args[0]):
		for tour in bbSubstitution(top, bbEdgesToBB(backbone)):
			reducedTours.add(tuple(tour))

	#put in a dictionary of tours with backbones
	smallest = {}
	for tour in reducedTours:
		smallest[tour] = tourLength(tour)

	if len(smallest) == 0:
		return tour
	else:
		return min(smallest, key=smallest.get)

def minmax(list):
    minVal = min(list)
    maxVal = max(list)

    return (minVal, maxVal)



#applies the backbone onto sections of a map
def windowMethod(cities, *args):

	altTour = list()
	density = 150

	#alter the cluster size depending on the size of the cities
	clusterSize = 10
	if len(cities) > 200:
		clusterSize = 20

	for cluster in generateClusters(set(cities), clusterSize, density):
		if len(cities) < 500:
			altTour.append(bbReduction(set(cluster), args[0]))
		else:
			#take a sample from cities and then calculate the average distance
			avgDis = 0
			count = 0
			citiesCopy = set(cities)
			if len(cities) > 10:
				temp = random.sample(cities, 1)[0]
				citiesCopy.remove(temp)
				while(count < int(round(len(cities)*0.1 + 0.5))):
					node = nearest_neighbor(temp, citiesCopy)
					avgDis += abs(temp - node)
					citiesCopy.remove(node)
					count += 1
			density = avgDis/int(round(len(cities)*0.1 + 0.5))

			# this will ignore smaller groups of nodes
			if len(cluster) >= 10:
				altTour.append(bbReduction(set(cluster), args[0]))
			else:
				altTour.append(set(cluster))

	cit = list()

	for tour in altTour:
		for city in tour:
			cit.append(city)

	if len(cities) > 1000:
		return repeatedNN(set(cit), 100)

	return 	repeatedNN(set(cit), len(cit))


#plot backboneclusters
def plotBackboneClusters(tour):
    #plotting backbones in the districts
    for cities in tour:
        if str(type(cities)) == "<class 'bbn.views.Backbone'>":
            plotTour(cities.y, "bb")
    plt.show()


#plot backbones

def plotBackbones(tour):
    #plots backbones individually
    for cities in tour:
        if str(type(cities)) == "<class 'bbn.views.Backbone'>":
            plotTour(cities.y, "bb")
            plt.show()

#need to save bbtour into variable to use thise
def plotPackage(bbtour):
	plotBackboneClusters(bbtour)
	plotBackbones(bbtour)


def info(bbtour, *args):
	temp = list()
	t0 = time.clock()
	bb = windowMethod(bbtour, args)
	t1 = time.clock()

	cartesian = list()
	listOfBackbones = list()

	for city in bb:
		if str(type(city)) == "<class 'bbn.views.Backbone'>":
			temp2 = list()
			for x in city.y:
				temp.append(x)
				cartesian.append(x.x)
				cartesian.append(x.y)
				temp2.append(x.x)
				temp2.append(x.y)
			listOfBackbones.append(temp2)
		else:
			temp.append(city)
			cartesian.append(city.x)
			cartesian.append(city.y)

	length = tourLength(temp)
	text = ("{} city tour with length {:.1f} in time {:.3f}".format(len(temp), length, t1-t0))
	print text

	return [cartesian, text, length, listOfBackbones]

def benchmarkBB(cartList):
	#go through the cartesian list and mark tuples
	listBB = []

	for xy in range(0, len(cartList) - 1, 2):
		listBB.append(complex(cartList[xy], cartList[xy + 1]))

	return listBB

def generateClusters(cities, clusterSize, radius):
	cluster = list()
	exit = False

	if len(cities) < 4:
		return cities

	while cities:
		tour = [random.choice(tuple(cities))]
		cities.remove(tour[-1])
		while len(tour) < clusterSize and exit == False and len(cities) > 0:
			node = nearest_neighbor(tour[len(tour) - 1], cities)
			if abs(node - tour[len(tour) - 1]) > radius:
				break
			tour.append(node)
			cities.remove(node)
		cluster.append(tour)
		exit = False
	return cluster


# rudimentary/redundant process to display backbone process on the landing page.
def landingBackbones(tours):
	t0 = time.clock()
	# conver the points to city objects
	tspSolutions = list()
	for i in range(6):
		tmp = list()
		tmp2 = tours[i*200: i*200 + 200]
		for j in range(len(tmp2)/2):
			tmp.append(City(tmp2[j*2], tmp2[j*2+1]))
		tspSolutions.append(tmp)

	#solutions into a dictionary by their length
	topDict = {}

	# calculates the length for each tour
	for tours in tspSolutions:
		topDict[tuple(tours)] = tourLength(tours)

	top = list()
	# gets top 10 percent of shortest tours, backbone error may get a backbone not in the tour???
	for i in range(0, 3):
		#picking method
		tmp = random.choice(topDict.keys())
		top.append(list(tmp))
		del topDict[tmp]


	# # looks for BB in all the tours in top
	# # substitute these backbones into all the tours
	reducedTours = set()
	for backbone in getAllBackBones(top, []):
		for tour in bbSubstitution(top, bbEdgesToBB(backbone)):
			reducedTours.add(tuple(tour))

	# # #put in a dictionary of tours with backbones
	smallest = {}
	for tour in reducedTours:
		smallest[tour] = tourLength(tour)

	cit = list(min(smallest, key=smallest.get))

	# # # nn with backbones
	result = repeatedNN(set(cit), len(cit) - 1)
	t1 = time.clock()

	cartesian = list()
	listOfBackbones = list()
	temp = list()

	for city in result:
		if str(type(city)) == "<class 'bbn.views.Backbone'>":
			temp2 = list()
			for x in city.y:
				temp.append(x)
				cartesian.append(x.x)
				cartesian.append(x.y)
				temp2.append(x.x)
				temp2.append(x.y)
			listOfBackbones.append(temp2)
		else:
			temp.append(city)
			cartesian.append(city.x)
			cartesian.append(city.y)

	length = tourLength(temp)
	text = ("{} city tour with length {:.1f} in time {:.3f}".format(len(temp), length, t1-t0))
	print text

	return [cartesian, text, length, listOfBackbones]

