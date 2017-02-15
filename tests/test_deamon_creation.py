from circusort.io import load_configuration
from circusort.deamon import Deamon


config = load_configuration()
deamons = [Deamon(host.name, username=host.username) for host in config.hosts]

print(deamons)
