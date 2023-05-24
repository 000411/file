from django import forms
from .models import Student


class StdForm(forms.Form):
    studentID = forms.IntegerField()
    name = forms.CharField(max_length=30)
    major = forms.CharField(max_length=100)