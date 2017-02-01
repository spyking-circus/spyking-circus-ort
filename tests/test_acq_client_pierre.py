import circusort
from circusort.config import Configuration
from circusort.acq import DataServerNode, DataReceiverNode
import socket

config   = Configuration('/home/pierre/data/sorting/GT_252/patch_2.params')


server   = DataServerNode(config)
receiver = DataReceiverNode(config)

if (socket.gethostbyname(socket.gethostname()) == config.acquisition.server_ip):
    server.run()
#else:
#    pass
#Here we are also listening on the same machine
receiver.run()