import circusort
from circusort.config import Configuration
from circusort.acq import DataServerNode, DataReceiverNode, RNGSource
from circusort.core import ZmqConnection
from circusort.devices import FilteringNode
import socket

config      = Configuration('/home/pierre/data/sorting/GT_252/patch_2.params')
is_master   = socket.gethostbyname(socket.gethostname()) == config.acquisition.server_ip
server      = DataServerNode()
receiver    = DataReceiverNode()


input_connection  = ZmqConnection(type='input', protocol='ipc', shape=(config.nb_channels, config.acquisition.buffer))
output_connection = ZmqConnection(type='output', protocol='ipc', shape=(config.nb_channels, config.acquisition.buffer))


#data_source       = ZmqConnection(type='input')
#data_source.connect(server)
#server.receive_from(data_source)


server.connect_to(output_connection)
receiver.receive_from(input_connection)

data_source = RNGSource(config)

if server.host == '127.0.1.1':
    server.set_data_source(data_source)
#else:
#    pass
#Here we are also listening on the same machine

input_connection_2  = ZmqConnection(type='input', protocol='ipc', shape=(config.nb_channels, config.acquisition.buffer))
output_connection_2 = ZmqConnection(type='output', protocol='ipc', shape=(config.nb_channels, config.acquisition.buffer))

filtering = FilteringNode()


server.connect_to(output_connection_2)
filtering.receive_from(input_connection_2)



receiver.start()
server.start()
filtering.start()

server = DataServerNode(interface="*") # read configuration locally
receiver = DataReceiverNode(interface="*") # read configuration locally
server = DataServerNode(interface="134.157.180.205", rpc_port=3426) # read configuration remotely
receiver = DataReceiverNode(interface="134.157.180.255", rpc_port=4732) # read configuration remotely
