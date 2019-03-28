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
from django.shortcuts import render, redirect, render_to_response
from django.http import JsonResponse
from django.conf import settings
from django.views.decorators.csrf import csrf_exempt

from .models import scoreboard
from .forms import registerForm

from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth.models import User
from django.contrib.sites.models import Site
from registration.models import RegistrationProfile

from django.contrib.staticfiles.storage import staticfiles_storage

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

def vsliresults(request):
	context = {}
	return render(request, 'vsli.html', context)

def graphOutput(request):
	context = {}
	return render(request, 'plot.html', context)

def getATour(request):

	citiesNum = request.GET.get('numCities', 10)
	backboneLen = request.GET.get('backboneLen', 0)
	newInstance = request.GET.get('newInstance', 0)

	if int(citiesNum) > 1000:
		print "Instance size of {:} is too large, size set to 1000.".format(
		    citiesNum)
		citiesNum = 1000

	# this takes a request from the front end and then the response is simple to render the html template
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

	# get the list of cities
	tourVal = tourVal.split(',')
	tourVal = map(float, tourVal)

	benchmarkBB(tourVal)

	# define variables for context
	tour = []
	tourLen = 0
	valid = False
	ratios = " "

	if len(tourKeyArr) == int(pathLen):
	 	valid = True

	# get the tour
	for i in tourKeyArr:
		tour.append(tourVal[i * 2])
		tour.append(tourVal[i * 2 + 1])

	# get the tour length
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

	for i in range(len(tourArr) / 2):
		tourCities.add(City(tourArr[i * 2], tourArr[i * 2 + 1]))

	tourBB = info(tourCities)

	context = {
		"tour": tourBB[0],
		"bbData": tourBB[1],
		"tourBBLen": tourBB[2],
		"backbones": tourBB[3],
		"lentour": len(tourArr) / 2,
	}

	return JsonResponse(context)

def calcDist(strTour):
	dist = 0
	for i in range(0, len(strTour) / 2):
		if i == (len(strTour) / 2 - 1):
			dist = dist + math.sqrt(abs(strTour[(i * 2)] - strTour[0])**2 +
									abs(strTour[(i*2) + 1] - strTour[1])**2)
		else:
			dist = dist + math.sqrt(abs(strTour[(i * 2)] - strTour[(i * 2) + 2])**2 +
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
	if request.user.is_active:
		if int(usrPathLen) == int(numNodes) and userLength != '0':
			tsp = tsp.split(',')
			tsp = map(float, tsp)

			tspKV = list()
			for i in range(0, int(numNodes)):
				tspKV.append((tsp[i * 2], tsp[i * 2 + 1], i))

			usrPath = str(usrPath)
			usrPath = usrPath[15:len(usrPath) - 7]

			newTupleScoreboard = scoreboard()
			newTupleScoreboard.id = request.user
			newTupleScoreboard.name = request.user.first_name + ' ' + request.user.last_name
			newTupleScoreboard.tourProblem = str(tspKV)
			newTupleScoreboard.userSolution = str(usrPath)
			newTupleScoreboard.userLength = userLength
			newTupleScoreboard.algorithmName = "Backbone Algorithm"
			newTupleScoreboard.algorithmLength = str(round(float(bbLen), 2))
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
		formz = registerForm(request.POST)
		users = User.objects.all()
		formEmail = formz['email'].value()
		usrName = formz['username'].value()
		for user in users:
			if str(user.username) == str(usrName):
				return render(request, 'registration/registration_form.html', {'form': formz, 'error': "Username already in use."})
			elif str(user.email) == str(formEmail):
				return render(request, 'registration/registration_form.html', {'form': formz, 'error': "Email already in use."})
		if formz.is_valid():
			print "registering account"
			mysite = Site.objects.get_current()
			usr = formz.save()
			usr.is_active = False
			usr.save()
			regProfile = RegistrationProfile.objects.create_profile(usr)
			regProfile.send_activation_email(mysite)
			return redirect('http://127.0.0.1:8000', args)
		return render(request, 'registration/registration_form.html', {'form': formz})
	else:
		form = registerForm()
		args = {'form': form}
		return render(request, 'registration/registration_form.html', args)

# Backend code

# generate all possible tours given a set of cities
possibletours = itertools.permutations

def alltours(cities):
	"""Return the shortest tour from a set of Cities."""
	return shortestTour(possibletours(cities))

def shortestTour(tours):
	"""Takes a set of tours and gets the smallest Tour."""
	return min(tours, key=tourLength)

def tourLength(tour):
	"""Return the length of a tour of complex points."""
	return sum(distance(tour[i - 1], tour[i]) for i in range(0, len(tour)))

class Point(complex):
	"""A object that represents a node."""
	x = property(lambda p: p.real)
	y = property(lambda p: p.imag)

City = Point

def distance(A, B, *args):
	"""Returns the distance between two Point object."""
	if str(type(B)) == "<class 'bbn.views.Backbone'>" and str(type(A)) == "<class 'bbn.views.Backbone'>":
		tmp = min(abs(A.y[0] - B.y[0]), abs(A.y[-1] - B.y[0]),
		          abs(A.y[0] - B.y[-1]), abs(A.y[-1] - B.y[-1]))
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

def firstElem(collection):
	"""Return first element in a collection."""
	return next(iter(collection))

def nearest_neighbor(city, cities):
	"""Returns a TSP solution using Nearest_neighbor approach."""
	minDist = min(cities, key=lambda b: distance(b, city))
	if str(type(minDist)) == "<class 'bbn.views.Backbone'>":
		if minDist.z == 'Lunsure':
			minDist.z = 'L'
		elif minDist.z == 'R':
			minDist.z = 'R'
	return minDist

def nnVariation(cities, start=None):
	"""Returns a TSP solution using nearest_neighbor starting at a specified point."""
	if start is None:
		start = firstElem(cities)

	tour = [start]
	unvisited = set(cities - {start})
	while unvisited:
		candidate = nearest_neighbor(tour[-1], unvisited)
		tour.append(candidate)
		unvisited.remove(candidate)

	for city in tour:
	    if str(type(city)) == "<class 'bbn.views.Backbone'>":
	        city.z = ' '

	return tour

def Cities(n, width=1000, height=600, seed=50):
	"""Returns a set of cities size n, saved to a seed between 0 - 50."""
	random.seed(seed * n)
	tempSet = set()
	for a in range(n):
		tempCity = City(random.randrange(width), random.randrange(height))
		while tempCity in tempSet:
			print "Finding a unique city ..."
			tempCity = City(random.randrange(width), random.randrange(height))
		tempSet.add(tempCity)
	return frozenset(tempSet)

def plotTour(tour, *args):
	"""Plots a tour of Points."""
	if not args:
		plotPoints(list(tour) + [tour[0]])
	else:
		plotPoints(list(tour))

def plotPoints(points):
	x = []
	y = []

	for p in points:
		if str(type(p)) == "<class 'bbn.views.Backbone'>":
			x.append(p.x.x)
			y.append(p.x.y)
		else:
			x.append(p.x)
			y.append(p.y)
	plt.plot(x, y, 'bo-')

def applyTSP(algo, cities):
	"""Returns a solution on the cities using the algorithm specified."""
	t0 = time.clock()
	tour = algo(cities)
	t1 = time.clock()
	print("{} city tour with length {:.1f} in {:.3f} secs for {}".format(
	    len(tour), tourLength(tour), t1 - t0, algo.__name__))
	plotTour(tour)
	plt.show()

def ratio(cities, cities2):
	"""Returns the difference between two solutions in ratio."""
	if (cities - cities2) == 0:
		return "Your tour is {:.1f} % better than the BB Tour".format(0)
	elif (cities - cities2) <= 0:
		return "Your tour is {:.3f}% worse than the BB Tour.".format((abs(cities - cities2) / cities))
	elif (cities - cities2) > 0:
		return "your tour is {:.3f}% better than the BB Tour".format((abs(cities - cities2) / cities))
	else:
		return "Invalid Tour Input"

def repeatedNN(cities, repititions=100):
	"""Returns a TSP solution using nearest_neighbor on 0 - 100 random cities."""
	return shortestTour(nnVariation(cities, start) for start in sample(cities, repititions))

def repeatedNN10(cities):
	"""Returns a TSP solution using nearest_neighbor on 10 random cities."""
	return repeatedNN(cities, 10)

def repeatedNN100(cities):
	"""Returns a TSP solution using nearest_neighbor on 100 random cities."""
	return repeatedNN(cities, 100)


def sample(cities, repititions, seed=42):
	"""Returns a sub-tour of cities from a list of cities."""
	return random.sample(cities, repititions)

def Maps(numMaps, numCities):
	"""Returns a map of unique TSP problems length numCities and number numMaps."""
	return tuple(Cities(numCities, seed=(n, numCities))
				 for n in range(numMaps))

def benchmark(funct, cities):
	"""Returns time taken to find a solution on cities using funct."""
	t0 = time.clock()
	results = [funct(x) for x in cities]
	t1 = time.clock()
	return (t1 - t0 / len(cities), results)

def benchmarks(functions, maps=Maps(30, 60)):
	"""Apples a list of functions onto a Map of TSP problems."""
	for function in functions:
		t, result = benchmark(function, maps)
		length = [tourLength(r) for r in result]
		print("{:>25} |{:7.0f} ±{:4.0f} ({:5.0f} to {:5.0f}) |{:7.3f} secs/map | {} ⨉ {}-city maps"
              .format(function.__name__, mean(lengths), stdev(lengths), min(lengths), max(lengths),
                      t, len(maps), len(maps[0])))

class Backbone(object):
	"""A Backbone Object that contains x and y corodinate and subset in z."""
	def __init__(self, x, y, *args):
		self.x = x
		self.y = y
		self.z = dict(args)

	def __repr__(self):
		return repr(self.x)

def getAllBackBones(tours, *args):
	"""Return a list of backbones from a set of tours greater than length 1."""
	# number of tours
	lengthTours = len(tours)

	# get first tour and its length
	firstTour = tours[0]
	lengthFirst = len(firstTour)

	# store backbones
	backbones = list()

	switch = False

	# all combinations
	for i in range(0, lengthFirst):
		for j in range(i + 1, lengthFirst + 1):
			tmp = firstTour[i:j]
			# loops through each tour to check if tmp is contained
			k = 1
			for k in range(k, lengthTours):
				tmpStr = ''.join(str(e) for e in tours[k])
				tmpTmp = ''.join(str(e) for e in tmp)
				# if subtour is not in any of the tours then we append the current bb
				if not (tmpTmp in tmpStr):
					if switch:
						backbones.append(firstTour[i:j - 1])
						switch = False
					break
				# we get to the end of the check of tours then we hit a inital match!
				if k == lengthTours - 1 and not switch:
					switch = True

	if len(args[0]) == 0:
		return prunebackBonesGELE(backbones, 2, 3)
	else:
		for x in args:
			return prunebackBonesLE(backbones, x[0])

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

def bbEdgesToBB(backboneEdges):
	"""Return a Backbone object from a list of Points."""
	real = []
	img = []

	for city in backboneEdges:
		real.append(city.x)
		img.append(city.y)

	return Backbone(City(round(sum(real) / len(backboneEdges), 0), round(sum(img) / len(backboneEdges), 0)), backboneEdges)

def bbSubstitution(collection, backbone):
	"""Returns the collection of tours with substituted backbones."""
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

def generateAllTours(cities):
	"""Returns all possible TSP solutions variations using nearest_neighbor."""
	alltours = list()
	if len(cities) < 2:
		return cities
	for city in cities:
		alltours.append(nnVariation(cities, city))
	return alltours

def bbReduction(tour, *args):
	"""Return the tour with a backbone substitution."""
	tspSolution = generateAllTours(tour)
	topDict = {}
	meanSqu = {}
	top = list()
	maxi = 0
	for tours in tspSolution:
		if str(type(tours)) == "<class 'bbn.views.Point'>":
			topDict[tuple([tours])] = 0
		else:
			if len(tours) < 2:
				topDict[tuple([tours])] = 0
			else:
				topDict[tuple(tours)] = tourLength(tours)

	if(len(tour) < 500):
		for i in range(0, int(round(len(tspSolution) * 0.6 + 0.5))):
			# Picking method
			tmp = random.choice(topDict.keys())
			top.append(list(tmp))
			del topDict[tmp]
	else:
		avg = sum(topDict.values()) / len(topDict)
		for val in topDict:
			meanSqu[val] = (topDict[val] - avg)**2
		standardDev = math.sqrt(sum(meanSqu.values()) / len(meanSqu))
		avg1 = avg + (standardDev / 2)
		store = {}
		candidate = 999999
		k = 0
		maxi = 0
		while(candidate > avg1 and k < 1000):
			#List of tours from dictionary of tours and calculates average
			top = list()
			tmp = random.sample(topDict, int(round(len(tspSolution) * 0.1 + 0.5)))
			tmp2 = 0
			for item in tmp:
				tmp2 += topDict[item]
				top.append(list(item))
			candidate = tmp2 / len(tmp)
			#Store these averages in a dictionary, K - average value, V - frequency
			if candidate < avg1:
				try:
					store[round(candidate, 2)].append(top)
					store[round(candidate, 2)][0] = store[round(candidate, 2)][0] + 1
				except:
					store[round(candidate, 2)] = [1, top]
			candidate = 999999999
			k += 1
		# Fitness Proportion Seletion
		indexRoulette = rouletteSel(store)
		counter = 0
		for item in store:
			if counter == indexRoulette:
				top = store[item][random.randint(1, len(store[item]) - 1)]
				break
			counter += 1

	backbones = list()
	for backbone in getAllBackBones(top, args[0]):
		backbones.append(bbEdgesToBB(backbone))

	reducedTours = set()
	for backbone in backbones:
		for tour in bbSubstitution(top, backbone):
			reducedTours.add(tuple(tour))
	smallest = {}
	for tour in reducedTours:
		smallest[tour] = tourLength(tour)
	if len(smallest) == 0:
		return [0, tour]
	else:
		return [maxi, min(smallest, key=smallest.get)]

def minmax(list):
	"""Return a tuple of smallest and largest values in list."""
	minVal = min(list)
	maxVal = max(list)
	return (minVal, maxVal)

def windowMethod(cities, *args):
	"""Return a TSP solution using Backbones."""
	altTour = list()
	density = 150
	avgDis = 0
	count = 0
	citiesCopy = set(cities)
	#Calculate density of clusters (avg distance)
	if len(cities) > 10:
		temp = random.sample(cities, 1)[0]
		citiesCopy.remove(temp)
		while(count < int(round(len(cities)*0.1 + 0.5))):
			node = nearest_neighbor(temp, citiesCopy)
			avgDis += abs(temp - node)
			citiesCopy.remove(node)
			count += 1
		density = avgDis/int(round(len(cities)*0.1 + 0.5))
	clusterSize = 10
	if len(cities) > 200:
		clusterSize = 100
	#Generate clusters
	for cluster in generateClusters(set(cities), clusterSize, density):
		if len(cities) < 1000:
			altTour.append(bbReduction(set(cluster), args[0]))
		else:
			if len(cluster) >= 80:
				altTour.append(bbReduction(set(cluster), args[0]))
			else:
				altTour.append([0, cluster])
	topSelection = 2
	cities = list()
	backbones = {}
	for i in range(len(altTour)):
		tmpVal = altTour[i][0]
		if (tmpVal > 0):
			backbones[i] = tmpVal
	if len(backbones) < 10:
		temp = list()
		for i in range(topSelection):
			if len(backbones) == 0:
				break
			maxIndex, maxValue = random.choice(list(backbones.items()))
			altTour[maxIndex][0] = -1
			del backbones[maxIndex]
			cities = cities + list(altTour[maxIndex][1])
	else:
		distances = False
		count = 0
		while not(distances):
			if len(backbones) == 0:
				break
			count += 1
			temp = list()
			indexes = list()
			# backbones is a KV pair, K - i, V - 0,n
			for choice in random.sample(list(backbones.items()), topSelection):
				indexes.append(choice[0])
				temp = temp + list(altTour[choice[0]][1])
			var = list()
			for item in temp:
				if str(type(item)) == "<class 'bbn.views.Backbone'>":
					var.append(item)
			if len(var) >= topSelection:
				breakout = False
				for vari in range(1, len(var)):
					if not distance(var[0], var[vari]) < density*2:
						breakout = True
						break
				if not breakout:
					distances = True
					cities = cities + temp
					for ind in indexes:
						altTour[ind][0] = -1
			if count == 100000:
				break

	for tour in altTour:
		if not tour[0] == -1:
			for city in tour[1]:
				if tour[0] > 0 and str(type(city)) == "<class 'bbn.views.Backbone'>":
					for cit in city.y:
						cities.append(cit)
				else:
					cities.append(city)

	if len(cities) < 200:
		return repeatedNN(set(cities), len(cities))
	else:
		return 	repeatedNN(set(cities), 100)

def rouletteSel(weightStore):
	"""Return selction from a list of tour lengths."""
	weights = list()
	sumOfWeights = 0
	for sample in weightStore:
		weights.append(weightStore[sample][0])
		sumOfWeights += weightStore[sample][0]
	selections = [0]*len(weights)
	for i in range(len(weights)):
		unlock = False
		value = random.uniform(0, 1)*sumOfWeights
		for i in range(len(weights)):
			value -= weights[i]
			if value < 0:
				selections[i] += 1
				break
			if (i == (len(weights) - 1)):
				unlock = True
		if unlock and value < 0:
			selections[len(weights) - 1] += 1
	return selections.index(max(selections))

def plotBackboneClusters(tour):
    """Plot all backbone in tour."""
    for cities in tour:
        if str(type(cities)) == "<class 'bbn.views.Backbone'>":
            plotTour(cities.y, "bb")
    plt.show()

def plotBackbones(tour):
    """Plot all backbone in tour on individual figures."""
    for cities in tour:
        if str(type(cities)) == "<class 'bbn.views.Backbone'>":
            plotTour(cities.y, "bb")
            plt.show()

def plotPackage(bbtour):
	plotBackboneClusters(bbtour)
	plotBackbones(bbtour)


def info(bbtour, *args):
	"""Main method."""
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
	"""Plot all backbone in tour."""
	listBB = []

	for xy in range(0, len(cartList) - 1, 2):
		listBB.append(complex(cartList[xy], cartList[xy + 1]))

	return listBB

def generateClusters(cities, clusterSize, radius):
	"""Returns number of cities of clusterSize that is in radius."""
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


def landingBackbones(tours):
	"""Main method."""
	t0 = time.clock()
	tspSolutions = list()
	for i in range(6):
		tmp = list()
		tmp2 = tours[i*200: i*200 + 200]
		for j in range(len(tmp2)/2):
			tmp.append(City(tmp2[j*2], tmp2[j*2+1]))
		tspSolutions.append(tmp)

	# solutions into a dictionary by their length
	topDict = {}

	# calculates the length for each tour
	for tours in tspSolutions:
		topDict[tuple(tours)] = tourLength(tours)

	top = list()
	# gets top 10 percent of shortest tours, backbone error may get a backbone not in the tour???
	for i in range(0, 3):
		# picking method
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
