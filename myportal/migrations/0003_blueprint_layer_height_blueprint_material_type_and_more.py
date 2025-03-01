# Generated by Django 5.1.1 on 2024-10-29 00:49

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('myportal', '0002_blueprint_blueprint_type_alter_blueprint_file'),
    ]

    operations = [
        migrations.AddField(
            model_name='blueprint',
            name='layer_height',
            field=models.DecimalField(blank=True, decimal_places=2, max_digits=5, null=True),
        ),
        migrations.AddField(
            model_name='blueprint',
            name='material_type',
            field=models.CharField(blank=True, max_length=50, null=True),
        ),
        migrations.AddField(
            model_name='blueprint',
            name='print_speed',
            field=models.DecimalField(blank=True, decimal_places=2, max_digits=5, null=True),
        ),
    ]
