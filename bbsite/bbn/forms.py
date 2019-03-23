from django import forms
from .models import scoreboard
from django.contrib.auth.models import User
from django.contrib.auth.forms import UserCreationForm

#inherit from UserCreationForm, and we overwrite parts of the class
class registerForm(UserCreationForm):
	email = forms.EmailField(required=True)

	class Meta:
		# setting which model to use for the form
		model = User
		# showing which fields in model to show
		fields = (
				 'username', 'first_name',
				 'last_name', 'email',
				 'password1', 'password2',
		)

	#save the information to the database, without first and last name won't save
	def save(self, commit=True):
		user = super(registerForm, self).save(commit=False)
		user.first_name = self.cleaned_data['first_name']
		user.last_name = self.cleaned_data['last_name']
		user.email = self.cleaned_data['email']

		if commit:
			user.save()

		return user
