from django import forms
from .models import Blueprint

class BlueprintForm(forms.ModelForm):
    user = forms.ModelChoiceField(queryset=User.objects.all(), required=True)
    class Meta:
        model = Blueprint
        fields = ['file', 'blueprint_type', 'print_speed', 'layer_height', 'material_type', 'user']
