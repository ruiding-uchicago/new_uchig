from django.db import models
from django.contrib.auth.models import User
import os

def blueprint_upload_to(instance, filename):
    return f'blueprints/{instance.blueprint_type}/{filename}'

class Blueprint(models.Model):
    BLUEPRINT_TYPES = [
        ('mechanical', 'Mechanical'),
        ('electrical', 'Electrical'),
        ('software', 'Software'),
        ('other', 'Other'),
    ]

    user = models.ForeignKey(User, on_delete=models.CASCADE)
    file = models.FileField(upload_to=blueprint_upload_to)
    blueprint_type = models.CharField(max_length=20, choices=BLUEPRINT_TYPES, default='other')
    uploaded_at = models.DateTimeField(auto_now_add=True)
    
    # Fields for printing parameters
    print_speed = models.DecimalField(max_digits=5, decimal_places=2, null=True, blank=True)
    layer_height = models.DecimalField(max_digits=5, decimal_places=2, null=True, blank=True)
    material_type = models.CharField(max_length=50, null=True, blank=True)

    def save(self, *args, **kwargs):
        super().save(*args, **kwargs)
        # Ensure the directory exists
        directory = os.path.dirname(self.file.path)
        if not os.path.exists(directory):
            os.makedirs(directory)
        # Write metadata to a file in the same directory as the uploaded file
        metadata_filename = os.path.splitext(self.file.path)[0] + '.metadata'
        with open(metadata_filename, 'w') as metadata_file:
            metadata_file.write(f"User: {self.user.username}\n")
            metadata_file.write(f"Blueprint Type: {self.get_blueprint_type_display()}\n")
            metadata_file.write(f"Print Speed: {self.print_speed}\n")
            metadata_file.write(f"Layer Height: {self.layer_height}\n")
            metadata_file.write(f"Material Type: {self.material_type}\n")
            metadata_file.write(f"Uploaded At: {self.uploaded_at}\n")

    def __str__(self):
        return f"{self.user.username} - {self.file.name}"