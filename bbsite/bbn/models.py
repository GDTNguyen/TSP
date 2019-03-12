# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.db import models
from django.conf import settings
from django.contrib.auth.models import User
import random
import datetime

# make FK to another model(primary_key)
class scoreboard(models.Model):
	id = models.ForeignKey(settings.AUTH_USER_MODEL, primary_key=True)
	name = models.CharField(max_length=123, blank=True, null=True)
	tourProblem = models.CharField(max_length=9999, blank=True, null=True)
	userSolution = models.CharField(max_length=999, blank=True, null=True)
	userLength = models.CharField(max_length=9999, blank=True, null=True)
	algorithmName = models.CharField(max_length=9999, blank=True, null=True)
	algorithmLength = models.CharField(max_length=9999, blank=True, null=True)
	dateSubmitted = models.DateField(default=datetime.date.today)
	numberOfNodes = models.CharField(max_length=9999, blank=True, null=True)

	def __unicode__(self):
		return self.name